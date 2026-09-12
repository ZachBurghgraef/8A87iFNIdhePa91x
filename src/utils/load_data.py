import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def get_split_from_file(filename:str, predictor: str, **kwargs) -> tuple:
    random_state = kwargs.get("random_state", 0)
    np.random.seed(random_state)
    test_size = kwargs["test_size"] if "test_size" in kwargs else 0.2
    df = pd.read_csv(filename)
    
    # format dataframe
    X = df
    Y = X[predictor]
    X = X.drop([predictor], axis=1)

    # Split into train and hold-out test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state, stratify=Y)
    return (X_train, X_test, y_train, y_test)
