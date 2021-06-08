from flask import Flask, request, jsonify
import os
import random
from keyword_spotting_service import KeywordSpottingService


app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():

    # get audio file and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)

    # invoke keyword spotting service
    kss = KeywordSpottingService()

    # make prediction
    predicted_keyword = kss.predict(file_name)

    # remove the audio file
    os.remove(file_name)

    # send back the predicted keyword as json
    data = {"keyword": predicted_keyword}

    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=False)
