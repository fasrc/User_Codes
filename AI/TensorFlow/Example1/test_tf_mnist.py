import pytest
import tensorflow as tf
import numpy as np
from tf_mnist import model, x_test, y_test, ROC_curves

def test_model_evaluation():
    score = model.evaluate(x_test, y_test, verbose=0)
    assert score[1] > 0.90, "Accuracy should be higher than 90%"

def test_predictions():
    predictions = model.predict(x_test)
    assert predictions.shape[0] == len(x_test), "Prediction shape mismatch"

def test_ROC_curves():
    predictions = model.predict(x_test)
    FPR, TPR, AUC = ROC_curves(y_test, predictions, 10)
    assert len(FPR) == 10, "FPR length mismatch"
    assert len(TPR) == 10, "TPR length mismatch"
    assert len(AUC) == 10, "AUC length mismatch"