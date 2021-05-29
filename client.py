from pathlib import Path
import requests
from read_config import read_config

conf = read_config("config.json")

URL = conf["URL"]
TEST_AUDIO_FILE_PATH = conf["TEST_AUDIO_FILE_PATH"]

if __name__ == "__main__":

    audio_file = open(Path(TEST_AUDIO_FILE_PATH), "rb")
    values = {"file": (TEST_AUDIO_FILE_PATH, audio_file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    print(f"Predicted keyword is: {data['keyword']}")