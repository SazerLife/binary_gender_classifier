from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import soundfile as sf
import torchaudio
from torchvision.transforms import Compose


class LibriTTS:
    def __load_dataset(
        self, data_path: Path, subset: str
    ) -> Tuple[List[Path], List[str]]:
        # "READER", "GENDER", "SUBSET", "NAME"
        speakers = pd.read_csv(data_path / "speakers.tsv", sep="\t")
        speakers = speakers.loc[speakers["SUBSET"] == subset]

        wav_paths: List[Path] = list()
        labels: List[str] = list()
        for reader, gender, _, _ in speakers.values:
            reader_dir: Path = data_path / subset / str(reader)
            reader_wav_paths = list(reader_dir.rglob("*.wav"))
            wav_paths.extend(reader_wav_paths)
            labels.extend([gender for _ in reader_wav_paths])

        return wav_paths, labels

    def __init_transforms(self, transform_configs: Dict):
        if not transform_configs:
            return None

        transforms = list()
        for transform_config in transform_configs:
            # Импорт класса трансформации
            module = import_module(transform_config["source"])
            transform_class = getattr(module, transform_config["name"])
            # Инициализация и добавление трансформации
            transform = transform_class(**transform_config["params"])
            transforms.append(transform)
        return Compose(transforms)

    def __init__(
        self,
        data_path: str,
        subset: str,
        audio_transform_configs=None,
        label_transform_configs=None,
    ):
        self.__wav_paths, self.__labels = self.__load_dataset(Path(data_path), subset)

        self.__audio_transform = self.__init_transforms(audio_transform_configs)
        self.__label_transform = self.__init_transforms(label_transform_configs)

    def __len__(self) -> int:
        return len(self.__labels)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # audio, sr = sf.read(self.__wav_paths[index])
        audio, sr = torchaudio.load(self.__wav_paths[index], normalize=True)
        label = self.__labels[index]

        if self.__audio_transform:
            data = self.__audio_transform({"audio": audio, "sr": sr})
            audio = data["audio"]
        if self.__label_transform:
            label = self.__label_transform(label)

        sample = {"audio": audio, "label": label}
        return sample
