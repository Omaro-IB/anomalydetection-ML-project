import train
from random import shuffle
from os import listdir
import pickle

def predictor(hourOfDay, dayOfWeek, monthOfYear, meter_reading, dir, n):
    """
    Uses random saved models in directory to make an anomaly prediction
    :param hourOfDay: Feature 1: hour of day
    :param dayOfWeek: Feature 2: day of week
    :param monthOfYear: Feature 3: month of year
    :param meter_reading: Feature 4: meter reading
    :param dir: Directory with .sav files of models (and only .sav files)
    :param n: Number of random .sav files of models to use
    :return: Float: 0 to 1 representing whether the features listed is an anomaly or not based on the random models' consensus
    """
    listOfFiles = listdir(dir)
    shuffle(listOfFiles)
    modelsToUse = listOfFiles[:n]

    votes = []
    for model in modelsToUse:
        loaded_model = pickle.load(open(dir+"\\"+model, 'rb'))
        votes.append(train.predict(loaded_model, [hourOfDay, dayOfWeek, monthOfYear, meter_reading]))

    return sum(votes) / len(votes)