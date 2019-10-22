import logging
from functools import partial

from typing import Sequence, Any, Iterable, Optional, List

import click
import click_log

import torch
import torch.nn as nn
import torch.nn.functional as tnnf
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

from ignite.metrics import Loss, Accuracy
from ignite.engine import Events, create_supervised_evaluator
from ignite.contrib.handlers.param_scheduler import LRScheduler

from libcrap import shuffled
from libcrap.torch import set_random_seeds
from libcrap.torch.click import (
    click_dataset_root_option, click_models_dir_option, click_tensorboard_log_dir_option,
    click_seed_and_device_options
)
from libcrap.torch.training import (
    add_checkpointing, add_early_stopping, add_weights_and_grads_logging,
    setup_trainer, setup_evaluator, setup_tensorboard_logger,
    make_standard_prepare_batch_with_events, add_logging_input_images
)

logger = logging.getLogger()
click_log.basic_config(logger)

MNIST_DATASET_SIZE = 60000
NUM_LABELS = 10

MNIST_TRANSFORM = transforms.Compose((
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.1,), (0.2752,))
))


h = 32
w = 32

h1 = 4
h2 = 4
hw = 4
w1 = 4
w2 = 4

o1 = 4
o2 = 4
o3 = 4
o4 = 4
o5 = 4

r1 = 8
r2 = 8
r3 = 8
r4 = 8


def permute_pixels(permutation: List[int], image: torch.Tensor) -> torch.Tensor:
    assert image.shape == (1, h, w)
    assert len(permutation) == h * w
    return image.reshape(h * w)[permutation].reshape(image.shape)


class TTMnistFirstLayer(nn.Module):
    def __init__(self, r1, r2, r3, r4):
        super().__init__()
        self.core1 = nn.Parameter(torch.randn(h1, r1, o1) * 0.8)
        self.core2 = nn.Parameter(torch.randn(h2, r1, r2, o2) * 0.1)
        self.core3 = nn.Parameter(torch.randn(hw, r2, r3, o3) * 0.1)
        self.core4 = nn.Parameter(torch.randn(w1, r3, r4, o4) * 0.1)
        self.core5 = nn.Parameter(torch.randn(w2, r4, o5) * 0.8)

    def forward(self, input_):
        reshaped_input = input_.reshape(-1, h1, h2, hw, w1, w2)
        
        # in the einsum below, n stands for index of sample in the batch,
        # abcde - indices corresponding to h1, h2, hw, w1, w2 modes
        # i, j, k, l - indices corresponding to the 4 tensor train ranks
        # v, w, x, y, z - indices corresponding to o1, o2, o3, o4, o5

        result = torch.einsum(
            "nabcde,aiv,bijw,cjkx,dkly,elz",
            reshaped_input, self.core1, self.core2, self.core3, self.core4, self.core5
        )
        return result.reshape(-1, o1*o2*o3*o4*o5)


class TTMnistModel(nn.Sequential):
    def __init__(self, tt_layer_args: Iterable[Any]):
        super().__init__(
            nn.Upsample(size=(h, w), mode="bilinear", align_corners=False),
            TTMnistFirstLayer(*tt_layer_args),
            nn.ReLU(),
            nn.Linear(o1*o2*o3*o4*o5, NUM_LABELS)
        )


@click.command()
@click_log.simple_verbosity_option(logger)
@click_dataset_root_option()
@click_models_dir_option()
@click_tensorboard_log_dir_option()
@click.option(
    "--train-dataset-size", "-t", type=click.IntRange(1, MNIST_DATASET_SIZE), default=58000,
)
@click.option(
    "--learning-rate", "-r", type=float, default=1e-2
)
@click.option(
    "--batch-size", "-b", type=int, default=100
)
@click.option("--shuffle-pixels", is_flag=True)
@click.option("--load-model", required=False, type=click.Path(exists=True, dir_okay=False))
@click.option("--train/--no-train", default=True)
@click.option("--test/--no-test", default=False)
@click_seed_and_device_options(default_device="cpu")
def main(
    dataset_root, train_dataset_size, tb_log_dir, models_dir,
    learning_rate, batch_size, device, seed, shuffle_pixels,
    load_model: Optional[str], train: bool, test: bool
):
    if not shuffle_pixels:
        transform = MNIST_TRANSFORM
    else:
        print("Pixel shuffling is enabled")
        pixel_shuffle_transform = transforms.Lambda(
            partial(permute_pixels, shuffled(range(h * w)))
        )
        transform = transforms.Compose((MNIST_TRANSFORM, pixel_shuffle_transform))
    model = TTMnistModel((r1, r2, r3, r4))
    if load_model is not None:
        model.load_state_dict(torch.load(load_model, "cpu"))
        logger.debug(f"Loaded model from {load_model}")
    metrics = {"cross_entropy_loss": Loss(tnnf.cross_entropy), "accuracy": Accuracy()}
    if train:
        dataset = MNIST(dataset_root, train=True, download=True, transform=transform)
        assert len(dataset) == MNIST_DATASET_SIZE
        train_dataset, val_dataset = random_split(
            dataset, (train_dataset_size, MNIST_DATASET_SIZE - train_dataset_size)
        )
        train_loader, val_loader = (
            DataLoader(
                dataset_, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda")
            )
            for dataset_ in (train_dataset, val_dataset)
        )
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, momentum=0.95, weight_decay=0.0005
        )

        prepare_batch_for_trainer = make_standard_prepare_batch_with_events(device)
        trainer = setup_trainer(
            model, optimizer, tnnf.cross_entropy, device=device,
            prepare_batch=prepare_batch_for_trainer
        )
        scheduler = LRScheduler(torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.8547
        ))
        trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)
        prepare_batch_for_val_evaluator = make_standard_prepare_batch_with_events(device)
        val_evaluator = setup_evaluator(
            model, trainer, val_loader, metrics, device=device,
            prepare_batch=prepare_batch_for_val_evaluator
        )
        checkpointer = add_checkpointing(
            models_dir, "cross_entropy_loss", val_evaluator,
            objects_to_save={"model": model}, model=model
        )
        add_early_stopping(
            trainer, val_evaluator, "cross_entropy_loss",
            patience_num_evaluations=25
        )
        with setup_tensorboard_logger(
                tb_log_dir, trainer, metrics.keys(), {"val": val_evaluator}, model=model
        ) as tb_logger:
            add_weights_and_grads_logging(trainer, tb_logger, model)
            add_logging_input_images(tb_logger, trainer, "train", prepare_batch_for_trainer)
            add_logging_input_images(
                tb_logger, val_evaluator, "val", prepare_batch_for_val_evaluator,
                another_engine=trainer
            )
            trainer.run(train_loader, max_epochs=100)
        if len(checkpointer._saved) > 0:
            best_model_path = checkpointer._saved[0][1][0]
            logger.info(f"The best model is saved at '{best_model_path}'")
            model.load_state_dict(torch.load(best_model_path))
    if test:
        test_dataset = MNIST(dataset_root, train=False, download=True, transform=transform)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == "cuda")
        )
        test_evaluator = create_supervised_evaluator(model, metrics, device)
        test_evaluator.run(test_loader)
        print(f"On test dataset the best model got: {test_evaluator.state.metrics}")


if __name__ == "__main__":
    main()
