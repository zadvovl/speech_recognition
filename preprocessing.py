from pathlib import Path
import librosa
import json
import time


start_time = time.time()

DATASET_PATH = r"C:\Users\User\speech_commands"
JSON_PATH = r"C:\Users\User\speech_commands\data.json"

# number of samples in a second
SAMPLE_RATE = 22050
# measured in seconds for the current dataset
DURATION = 1
# could have just set it to 22050, but let's make it more generic
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, is_test=True):

    # dictionary to store data
    data = {
        "mapping" : [],
        "mfcc" : [],
        "labels" : [],
        "files" : []
    }

    p = Path(dataset_path)

    # creating a list with labels from child folder names
    labels = []
    for d in p.iterdir():
        if d.is_dir():
            labels.append(d.name)

    # saving a list of labels in a dictionary
    data["mapping"] = labels

    # find all wav files in DATASET_PATH
    all_files = p.glob("**/*.wav")

    # loop through all the wav files and extract required metadata
    for f in all_files:

        # save the semantic label
        semantic_label = f.parent.name

        # process file itself
        signal, sr = librosa.load(f, sr=SAMPLE_RATE)

        # let's make sure that the file is at least 1 second long
        if len(signal) >= SAMPLES_PER_TRACK:

            # enforce 1 second long signal
            signal = signal[:SAMPLES_PER_TRACK]

            # extract mfcc, store the data
            mfcc = librosa.feature.mfcc(signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

            # transposing because it's easier to work with MFCCs this way (otherwise coefficient lists are in rows
            # which is not really convenient)
            mfcc = mfcc.T

            # store mfcc (casting to list because ndarray can't be stored in a json file)
            data["mfcc"].append(mfcc.tolist())

            # just searching for an index value that corresponds to genre label
            data["labels"].append(labels.index(semantic_label))

            # saving file name
            data["files"].append(str(f.absolute()))

        # if it's a test run then only one file will be processed and the program will break out of the loop
        if is_test:
            print('This is a test run. Stopping here.')
            break

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, is_test=False)

    print(f"Program took {(time.time() - start_time)} seconds to execute")