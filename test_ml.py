import pytest
# TODO: add necessary import
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference


# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """
    Ensure that the training function returns a trained
    RandomForestClassifier object.
    """
    # Your code here
    X_sample = np.random.rand(25, 6)
    y_sample = np.random.randint(0, 2, 25)
    trained_model = train_model(X_sample, y_sample)
    assert isinstance(trained_model, RandomForestClassifier)


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    Confirm that precision, recall, and F1 score
    are returned as numeric float values.
    """
    # Your code here
    actual = np.array([1, 0, 1, 0, 1])
    predicted = np.array([1, 0, 0, 0, 1])
    precision, recall, f1 = compute_model_metrics(actual, predicted)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    Verify that inference returns the same number
    of predictions as rows in the input data.
    """
    # Your code here
    X_data = np.random.rand(18, 4)
    y_data = np.random.randint(0, 2, 18)
    model = train_model(X_data, y_data)
    predictions = inference(model, X_data)
    assert len(predictions) == X_data.shape[0]
