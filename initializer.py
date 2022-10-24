from typing import Dict, Tuple, Any
import logging
from importlib import import_module

import torch

LOGGER = logging.getLogger(__name__)


class Initializer:
    def __init__(self, config: Dict):
        self._train_config = config["train"].copy()
        self._val_config = config["val"].copy()
        self._train_objects_config = config["training_objects"].copy()
        self._metrics_config = config["metrics"].copy()
        self._loggers_config = config["loggers"].copy()

        self.__device = config["meta"]["device"]
        self.__meta_config = config["meta"].copy()

    @property
    def meta_config(self):
        return self.__meta_config

    def __init_class(self, config: Dict[str, Any]):
        module = import_module(config["source"])
        Class = getattr(module, config["name"])
        return Class(**config["params"])

    def init_data(self, data_name: str):
        config = getattr(self, f"_{data_name}_config")
        dataset = self.__init_class(config["dataclass"])
        dataloader_config = config["dataloader"].copy()
        collator = self.__init_class(dataloader_config["collator"])
        dataloader_config["params"].update(dict(dataset=dataset, collate_fn=collator))
        dataloader = self.__init_class(dataloader_config)

        return dataloader, collator

    def init_metrics(self):
        metrics = dict()
        for metric_name, metric_config in self._metrics_config.items():
            metric = self.__init_class(metric_config)
            metrics[metric_name] = metric
        return metrics

    def init_loggers(self):
        loggers = dict()
        for logger_name, logger_config in self._loggers_config.items():
            logger = self.__init_class(logger_config)
            loggers[logger_name] = logger
        return loggers

    def init_training_objects(self) -> Tuple[torch.nn.Module]:
        model_config = self._train_objects_config["model"].copy()
        model = self.__init_class(model_config)

        loss_config = self._train_objects_config["loss"].copy()
        loss = self.__init_class(loss_config)

        optimizer_config = self._train_objects_config["optimizer"].copy()
        optimizer_config["params"]["params"] = model.parameters()
        optimizer = self.__init_class(optimizer_config)

        epoch = 0
        iteration = 0
        checkpoint_path = self.__meta_config["checkpoint_path"]

        if checkpoint_path:
            checkpoint_dict = torch.load(checkpoint_path)
            LOGGER.info(f"Loaded checkpoint from {checkpoint_path}")

            if self._train_config["model"]["use_checkpoint"]:
                model.load_state_dict(checkpoint_dict["state_dict"])

            if self._train_config["optimizer"]["use_checkpoint"]:
                if not checkpoint_dict["optimizer"]:
                    raise RuntimeError("Checkpoint doesn't have optimizer data")

                optimizer.load_state_dict(checkpoint_dict["optimizer"])
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(self.__meta_config["device"])

            if self.__meta_config["load_checkpoint_epoch"]:
                epoch = checkpoint_dict["epoch"]

            if self.__meta_config["load_checkpoint_iteration"]:
                iteration = checkpoint_dict["iteration"]

        return (
            model.to(self.__device),
            optimizer,
            loss,
            epoch,
            iteration,
        )
