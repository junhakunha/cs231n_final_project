Change HOME_DIR in `src/utils/constants.py` to fit your home directory.

Download MNIST and construct weakly supervised dataset:
```
python -m data.mnist.download_MNIST
```

Training scripts
```
python -m src.training.train_supervised
python -m src.training.train_siamese
python -m src.training.train_weakly_supervised
```