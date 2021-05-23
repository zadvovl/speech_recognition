from pathlib import Path
import librosa
import json
import multiprocessing as mp
import itertools
import time


start_time = time.time()

DATASET_PATH = r"C:\Users\User\speech_commands"
JSON_PATH = r"C:\Users\User\speech_commands\data1.json"

# number of samples in a second
SAMPLE_RATE = 22050
# measured in seconds for the current dataset
DURATION = 1
# could have just set it to 22050, but let's make it more generic
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


# a function that will process files in a multithreaded way
def process_file(data, f, lock, n_mfcc=13, n_fft=2048, hop_length=512):
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

        # acquiring a lock to preserve shared resources from being written to until the process is done
        lock.acquire()

        # store mfcc (casting to list because ndarray can't be stored in a json file)
        data["mfcc"].append(mfcc.tolist())

        # just searching for an index value that corresponds to genre label
        data["labels"].append(data["mapping"].index(semantic_label))

        # saving file name
        data["files"].append(str(f.absolute()))

        # releasing the lock so that other processes can now use shared resources
        lock.release()


def save_mfcc(dataset_path):

    p = Path(dataset_path)

    # find all wav files in DATASET_PATH
    all_files = list(p.glob("**/*.wav"))

    with mp.Manager() as manager:

        # dictionary to store data (shared object)
        data = manager.dict({
            "mapping": manager.list(),
            "mfcc": manager.list(),
            "labels": manager.list(),
            "files": manager.list()
        })

        # creating a list with labels from child folder names
        labels = []
        for d in p.iterdir():
            if d.is_dir():
                labels.append(d.name)

        # saving a list of labels in a dictionary
        data["mapping"] = labels

        # we need the lock inside of process_file function to keep a correct sequence when appending lists (so that
        # a process working on a differnt file does not append to the list until all the values for the current
        # file are in place)
        l = manager.Lock()

        with manager.Pool() as pool:
            pool.starmap(process_file, zip(itertools.repeat(data), all_files, itertools.repeat(l)))

        # performing copy of the data dictionary as well as proxy lists from within the dictionary because otherwise
        # proxy objects (DictProxy and ListProxy) won't be serialized to json and the program will end with an error
        result = data.copy()
        result["mfcc"] = data["mfcc"][:]
        result["labels"] = data["labels"][:]
        result["files"] = data["files"][:]

        return result


if __name__ == "__main__":

    result = save_mfcc(DATASET_PATH)

    with open(JSON_PATH, "w") as fp:
        json.dump(result, fp, indent=4)

    print(f"Program took {(time.time() - start_time)} seconds to execute")