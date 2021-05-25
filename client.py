from pathlib import Path
import requests

URL = "http://127.0.0.1:5000/predict"
TEST_AUDIO_FILE_PATH = "./test/left.wav"

if __name__ == "__main__":

    audio_file = open(Path("./test/down.wav"), "rb")
    values = {"file": (TEST_AUDIO_FILE_PATH, audio_file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    print(f"Predicted keyword is: {data['keyword']}")