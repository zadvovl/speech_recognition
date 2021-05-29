import tensorflow.keras as keras
import numpy as np
import librosa
from pathlib import Path
from read_config import read_config

conf = read_config("config.json")

SAVED_MODEL_PATH = "model.h5"
SAMPLES_PER_TRACK = 22050

class _Keyword_Spotting_Service:
    # It will be a singleton (a class that can only have 1 instance in a program).

    model = None
    _mappings = [
        "down",
        "go",
        "left",
        "no",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes"
    ]
    _instance = None # we need this one because Python doesn't enforce singleton

    def predict(self, file_path):

        # extract MFCCs
        mfccs = self.preprocess(file_path) # (# segments, # coefficients)

        # convert 2d MFCC array into 4d array # -> (# samples, # segments, # coefficients, # channels = 1 in this case)
        mfccs = mfccs[np.newaxis, ..., np.newaxis]

        # make prediction
        predictions = self.model.predict(mfccs) # [ [0.1, 0.6, ..., 0.05, 0.1] ]
        predicted_index = np.argmax(predictions) # returns the index of the highest probability from predictions
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword


    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):

        # load_audio file
        signal, sr = librosa.load(file_path)

        # ensure that audio file length is consistent
        if len(signal) > SAMPLES_PER_TRACK:
            signal = signal[:SAMPLES_PER_TRACK]

        # extract MFCCs
        mfccs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        return mfccs.T


def KeywordSpottingService():

    # ensure that we only have one instance of _Keyword_Spotting_Service
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(SAVED_MODEL_PATH)

    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":

    kss = KeywordSpottingService()

    keyword1 = kss.predict(Path("./test/down.wav"))
    keyword2 = kss.predict(Path("./test/left.wav"))

    print(f"Predicted keywords: {keyword1}, {keyword2}")

