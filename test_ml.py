import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.model import train_model, compute_model_metrics, inference


# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    Test that train_model returns a RandomForestClassifier instance.
    """
    # Your code here
    X = np.random.rand(20, 5)
    y = np.random.randint(0, 2, 20)

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    Test that compute_model_metrics returns float values.
    """
    # Your code here
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])

    precision, recall, f1 = compute_model_metrics(y_true, y_pred)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    Test that inference returns predictions with correct length.
    """
    # Your code here
    X = np.random.rand(15, 4)
    y = np.random.randint(0, 2, 15)

    model = train_model(X, y)
    preds = inference(model, X)

    assert len(preds) == len(X)
