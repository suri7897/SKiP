# datasets/download.py

import os
import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer, load_iris
def save_npz(path, X, y):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    np.savez(path, X_train=X, y_train=y)
    print(f"\nSaved: {path}")

    with np.load(path) as tmp:
        print("Check:", tmp["X_train"].shape, tmp["y_train"].shape, " | unique labels:", len(np.unique(tmp["y_train"])))


def download_wine(save_dir):
    wine = load_wine()
    X, y = wine.data, wine.target
    path = os.path.join(save_dir, "wine", "wine.npz")
    save_npz(path, X, y)


def download_breast_cancer(save_dir):
    data = load_breast_cancer()
    X, y = data.data, data.target
    path = os.path.join(save_dir, "breast_cancer", "breast_cancer.npz")
    save_npz(path, X, y)


def download_iris_2feat(save_dir):
    iris = load_iris()
    X = iris.data[:, :2]   # only first two features
    y = iris.target
    path = os.path.join(save_dir, "iris_2feat", "iris_2feat.npz")
    save_npz(path, X, y)

def main():
    save_dir = "."
    os.makedirs(save_dir, exist_ok=True)

    download_wine(save_dir)
    download_breast_cancer(save_dir)
    download_iris_2feat(save_dir)


if __name__ == "__main__":
    main()
