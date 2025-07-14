import pathlib as pl
from sklearn.datasets import make_classification
import pandas as pd


def generate_fake_raw_dummy_data(path: pl.Path):
    X, y = make_classification()
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target").to_frame()
    df = pd.concat([X, y], axis=1)
    df.to_csv(path)
