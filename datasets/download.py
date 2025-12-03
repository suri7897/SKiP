# datasets/download.py

import os
import numpy as np
import pandas as pd
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


def download_titanic(save_dir):
    # Download Titanic dataset from direct URL
    import tempfile
    import zipfile
    import urllib.request
    
    # URL for Titanic dataset (Kaggle)
    url = "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/3136/26502/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1764787287&Signature=WvfevLPqaog3TKaMT%2FGmNBnlye%2BcMEsZqroon4Q4t9K4fV6q2Nz1JP0U985KjvudQhzmu%2FHpYZ0CGaUULl1GZ4oVAgK%2BTgZ2MsOtU1qvnvzdDlfhtGXrqHpU15DcdzXbSQIwymrgqgUNebRKQwmQayytx6jIUw4BhAvHDByT%2B3UCd2sNA7YcELwI0N5Y386uctU6QIsCrYLxaOoK%2F%2B8AaoNPYo8A%2BqohHGHgVZJN77I7VP17KtzvFe0Qam1doAorP%2FW%2Bd4KuXoemmCK0xqhW0EvRIwQTQ6f2ojDY3O5o0ni9PGeiPEV4Fl2v0NmEOzVKNo1pBGf9qw6l0QZevXdx3g%3D%3D&response-content-disposition=attachment%3B+filename%3Dtitanic.zip"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download dataset from URL
        # print("Downloading Titanic dataset from URL...")
        zip_path = os.path.join(tmpdir, "titanic.zip")
        urllib.request.urlretrieve(url, zip_path)
        
        # Unzip the downloaded file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # Load train.csv
        train_path = os.path.join(tmpdir, "train.csv")
        df = pd.read_csv(train_path)
    
    # === Preprocess the data ===
    # Select target variable (Survived)
    y = df['Survived'].values
    
    # Select features and handle missing values
    # Use numerical features: Pclass, Age, SibSp, Parch, Fare
    # Use categorical features: Sex, Embarked
    # Unused features: PassengerId, Name, Ticket, Cabin (not useful or too many missing)
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df_features = df[features].copy()
    
    # Fill missing values
    df_features['Age'] = df_features['Age'].fillna(df_features['Age'].median())
    df_features['Fare'] = df_features['Fare'].fillna(df_features['Fare'].median())
    df_features['Embarked'] = df_features['Embarked'].fillna(df_features['Embarked'].mode()[0])
    
    # Encode categorical variables (One-hot encoding)
    df_features = pd.get_dummies(df_features, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Convert to numpy array
    X = df_features.values.astype(float)
    
    # Remove rows with any remaining NaN values
    valid_idx = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]
    
    path = os.path.join(save_dir, "titanic", "titanic.npz")
    save_npz(path, X, y)


def main():
    save_dir = "."
    os.makedirs(save_dir, exist_ok=True)

    download_wine(save_dir)
    download_breast_cancer(save_dir)
    download_iris_2feat(save_dir)
    download_titanic(save_dir)


if __name__ == "__main__":
    main()
