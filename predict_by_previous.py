import sys
import numpy as np
from manager import KeyStrokeManager
import csv
from sklearn.externals import joblib


def predict_by_previous(data):

    modelPath = KeyStrokeManager.m_path + data[0]['user'] + '.pkl'
    paramPath = KeyStrokeManager.m_path + data[0]['user'] + '_params.csv'

    keyStrokeManager = KeyStrokeManager()
    normalizedFeatures = keyStrokeManager.parseTestFeatures(data, paramPath)
    model = joblib.load(modelPath)
    predictions = model.predict(normalizedFeatures)
    results = [KeyStrokeManager.invEmotions[x] for x in predictions]

    return results
