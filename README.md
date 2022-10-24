# Binary gender classifier
Репозиторий-решение для задачи классификации пола на датасете LibtiTTS

## Quick Start
* Установите [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation)

* Создайте новое виртуальное окружение
```bash
conda create -n gender_clf python=3.9
```

* И активируйте его
```bash
conda activate gender_clf
```

* Установите [PyTorch](https://pytorch.org/) (`torch` `torchvision` `torchaudio`)

* Установите необходимые зависимости
```bash
pip install -r requirements.txt
```


## Вопроизведение экспериментов

### Загрузка данных
* Установите wget последней версии (optional, вы можете скачать датасет любым удобным способом)
```bash
sudo apt install wget
```
* Создайте дирректорию для данных и скачайте туда датасет
```bash
wget -P data/external/ https://www.openslr.org/resources/60/train-clean-100.tar.gz;
wget -P data/external/ https://www.openslr.org/resources/60/dev-clean.tar.gz;
```
* Распакуйте данные в общую папку `data/external/LibriTTS`. В конечном счёте дерево данных должно быть таким:
```
├── data
│   ├── external
│   │   ├── LibriTTS
│   │   │   ├── dev-clean
│   │   │   │   ├── 84
│   │   │   │   ├── ...
│   │   │   │   └── 8842
│   │   │   ├── train-clean-100
│   │   │   │   ├── 19
│   │   │   │   ├── ...
│   │   │   │   └── 8975
│   │   │   └── speakers.tsv
```
Возможно, что в `data/external/LibriTTS/speakers.tsv` испортились tab-ы в заголовке, поправьте их, если это так.

### Запуск эксперимента
* Запустите эксперимент. Например, последний:
```bash
python train.py -c experiments/exp3-melspectrogram-resnet18/config.yaml
```

* Отслеживайте метрики:
```bash
tensorboard --logdir experiments/exp3-melspectrogram-resnet18/tensorboard/
```

## Citation
В данной работе используются наборы данных из датасета LibriTTS
```
@inproceedings{korvas_2014,
  title={{Free English and Czech telephone speech corpus shared under the CC-BY-SA 3.0 license}},
  author={Korvas, Mat\v{e}j and Pl\'{a}tek, Ond\v{r}ej and Du\v{s}ek, Ond\v{r}ej and \v{Z}ilka, Luk\'{a}\v{s} and Jur\v{c}\'{i}\v{c}ek, Filip},
  booktitle={Proceedings of the Eigth International Conference on Language Resources and Evaluation (LREC 2014)},
  pages={To Appear},
  year={2014},
}
```