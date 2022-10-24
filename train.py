import random
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from bestconfig import Config
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from initializer import Initializer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Скрипт для запуска обучения")
    parser.add_argument(
        "-c", "--config_path", required=True, type=Path, help="Путь до конфига"
    )
    return parser


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loss_fn,
    device,
    dataloader,
    collator,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    model.eval()
    batched_outputs = list()
    batched_targets = list()
    loss_values: List[float] = list()
    for batch in tqdm(dataloader):
        x, y = collator.x_y_split(batch)
        x = collator.x_to_device(x, device)
        y = collator.y_to_device(y, device)
        y_pred = model(x)
        loss_val = loss_fn(y_pred, y)

        batched_outputs.append(collator.y_to_device(y_pred, "cpu"))
        batched_targets.append(collator.y_to_device(y, "cpu"))
        loss_values.append(loss_val.item())

    return batched_outputs, batched_targets, loss_values


@torch.no_grad()
def eval_metrics(
    metrics: Dict[str, Any],
    batched_outputs: List[Dict[str, Any]],
    batched_targets: List[Dict[str, Any]],
) -> dict[str, float]:
    metrics_values = dict[str, float]()
    for metric_name, metric_fn in metrics.items():
        value = metric_fn(batched_outputs, batched_targets)
        metrics_values[metric_name] = value

    return metrics_values


def train(config: Dict):
    initializer = Initializer(config)

    seed = initializer.meta_config["seed"]
    device = initializer.meta_config["device"]
    do_clip_grad_norm = initializer.meta_config["do_clip_grad_norm"]
    grad_clip_threshold = initializer.meta_config["grad_clip_threshold"]
    evaluations_step = initializer.meta_config["evaluations_step"]
    max_epochs_num = initializer.meta_config["max_epochs_num"]

    seed_everything(seed)
    training_objects = initializer.init_training_objects()
    model, optimizer, loss, epoch, iteration = training_objects
    train_dataloader, train_collator = initializer.init_data("train")
    val_dataloader, val_collator = initializer.init_data("val")
    metrics: Dict[str, Any] = initializer.init_metrics()
    loggers: Dict[str, Any] = initializer.init_loggers()

    while True:
        model.train()
        batched_outputs = list()
        batched_targets = list()
        loss_values = list()
        pg_bar = tqdm(train_dataloader, desc=f"Batches at {epoch} epoch")
        for batch in pg_bar:
            train_iteration_start_time = time.perf_counter()

            x, y = train_collator.x_y_split(batch)
            x = train_collator.x_to_device(x, device)
            y = train_collator.y_to_device(y, device)
            y_pred = model(x)
            loss_val = loss(y_pred, y)
            train_reduced_loss = loss_val.item()
            loss_val.backward()

            if do_clip_grad_norm:
                grad_norm = clip_grad_norm_(model.parameters(), grad_clip_threshold)

            optimizer.step()
            model.zero_grad(set_to_none=True)

            with torch.no_grad():
                batched_outputs.append(train_collator.y_to_device(y_pred, "cpu"))
                batched_targets.append(train_collator.y_to_device(y, "cpu"))
                loss_values.append(train_reduced_loss)

            iteration += 1
            lr = optimizer.param_groups[0]["lr"]
            train_iteration_duration = time.perf_counter() - train_iteration_start_time

            pg_bar.write(
                f"\nTrain: Epoch: {epoch} | Iteration: {iteration} | Learning rate: {lr:.6} | Loss: {train_reduced_loss:.6f} | Grad Norm {grad_norm:.6f} | {train_iteration_duration:.2f}s/it"
            )

            if iteration % evaluations_step == 0:
                training_objects = (model, optimizer, epoch, iteration)
                train_data = batched_outputs, batched_targets, loss_values
                val_data = validate(model, loss, device, val_dataloader, val_collator)
                train_metrics_values = eval_metrics(
                    metrics, train_data[0], train_data[1]
                )
                val_metrics_values = eval_metrics(metrics, val_data[0], val_data[1])

                for logger in loggers.values():
                    logger.logging(
                        training_objects,
                        train_data,
                        val_data,
                        train_metrics_values,
                        val_metrics_values,
                    )
                model.train()

        epoch += 1
        if epoch == max_epochs_num:
            pg_bar.write(
                f"\nFinishing training due to maximum epochs limit: {max_epochs_num}"
            )
            break
        pg_bar.set_description(f"Batches at {epoch} epoch")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = Config(args.config_path, exclude_default=True)
    train(dict(config))
