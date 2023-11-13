# L3F-Net

A PyTorch implementation of _[TIP 2021 paper](https://ieeexplore.ieee.org/document/9305989), "Harnessing Multi-View Perspective of Light Fields for Low-Light Imaging"_

> This repo is _NOT_ the official implementation of the paper. For the official implementation, visit [here](https://mohitlamba94.github.io/L3Fnet/).

## Quick Start

First, download the [L3F dataset](https://mohitlamba94.github.io/L3Fnet/) and symlink it to the project root. The project directory structure should resemble:

```
|-- L3FNet
    |-- L3F-dataset
        |-- jpeg
            |-- train
            |-- test
    |-- train.py
    |-- eval.py
    ...
```

Modify the `split` item in `config.toml` to specify the subset for training/evaluation/testing:

``````
[train.dataset]
split = '20'   # subset for training, should be '20', '50' or '100'

[val.dataset]
split = '20'  # subset for eval

[test.dataset]
split = '20'  # subset for testing
``````

To start training or testing, execute the following commands:

```sh
# training
python train.py config.toml

# testing
python eval.py config.toml --ckpt ${CKPT_PATH}
```
