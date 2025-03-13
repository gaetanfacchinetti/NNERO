import pytest
import nnero

def test_load_classifier():
    assert nnero.Classifier.load()

def test_load_regressor():
    assert nnero.Regressor.load()

def test_predict_classifier():
    assert (nnero.predict_classifier() is not None)

def test_predict_regressor():
    assert (nnero.predict_Xe() is not None)
