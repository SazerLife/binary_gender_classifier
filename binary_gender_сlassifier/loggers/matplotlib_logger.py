from typing import Dict, Iterable, List, Tuple, Any
from pathlib import Path

import matplotlib.pyplot as plt
from torch import Tensor, nn


class MatplotlibLogger:
    def __init__(self, folder_to_save: str):
        self.__folder_to_save = Path(folder_to_save)
        self.__folder_to_save.mkdir(parents=True, exist_ok=True)

    @property
    def every_epoch(self):
        return self.__every_epoch

    def logging(
        self,
        training_objects: Tuple[nn.Module, nn.Module, int, int],
        train_data,
        val_data,
        train_metrics_values: Dict[str, Any],
        val_metrics_values: Dict[str, Any],
    ) -> None:
        _, _, epoch, iteration = training_objects
        _, _, train_loss_values = train_data
        _, _, val_loss_values = val_data

        # mean_losses = sum(train_loss_values) / len(train_loss_values)
        plt.plot(sum(train_loss_values) / len(train_loss_values))
        plt.plot(sum(val_loss_values) / len(val_loss_values))
        plt.savefig(self.__folder_to_save / f"{epoch}ep_{iteration}iter_loss.png")
        plt.close()

        for metric_name, train_metric, val_metric in zip(
            train_metrics_values.keys(),
            train_metrics_values.values(),
            val_metrics_values.values(),
        ):
            if not hasattr(self, metric_name):
                setattr(self, metric_name, list())
                metric: List[float] = getattr(self, metric_name)
                metric.append(metrics[metric_name].detach().cpu().item())
            plt.plot(metric)
            plt.savefig(
                f"{self.__folder_to_save}/{epoch}ep_{iteration}iter_{metric_name}.png"
            )
            plt.close()
