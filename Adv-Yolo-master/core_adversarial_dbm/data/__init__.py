import numpy as np
from joblib import Memory
from sklearn import datasets
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch as T
from core_adversarial_dbm.defs import ROOT_PATH
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


def load_mnist():
    memory = Memory(ROOT_PATH / "tmp")
    fetch_openml_cached = memory.cache(datasets.fetch_openml)

    X, y = fetch_openml_cached(
        "mnist_784",
        return_X_y=True,
        cache=True,
        as_frame=False,
    )
    return X, y


def load_fashionmnist():
    memory = Memory(ROOT_PATH / "tmp")
    fetch_openml_cached = memory.cache(datasets.fetch_openml)

    X, y = fetch_openml_cached(
        "Fashion-MNIST", return_X_y=True, cache=True, as_frame=False
    )
    return X, y


# def load_cifar10():
#     cifar10 = CIFAR10("/tmp", download=True, transform=ToTensor())

#     X = cifar10.data[...,0]
#     X = X.reshape(X.shape[0], -1)

#     return X, cifar10.targets

# def load_cifar10(flatten: bool = True):
#     cifar10 = CIFAR10("/tmp", download=True)

#     # 全部图像数据，形状为 (50000, 32, 32, 3)
#     X = cifar10.data

#     # 若需要展开为 (N, 3072) 形式用于 t-SNE 或神经网络输入
#     if flatten:
#         X = X.reshape(X.shape[0], -1)


#     return X, cifar10.targets

def load_cifar10():
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    cifar10 = CIFAR10("/tmp", download=True, transform=transform)
    X = T.stack([img for img, _ in cifar10])
    y = T.tensor([label for _, label in cifar10])
    return X, y


def load_quickdraw():
    base_path = ROOT_PATH / "data" / "assets" / "quickdraw"
    X = np.load(base_path / "X.npy")
    y = np.load(base_path / "y.npy")
    return X, y
