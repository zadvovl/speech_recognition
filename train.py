import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras


DATA_PATH = r"C:\Users\User\speech_commands\data1.json"
SAVED_MODEL_PATH = r"model.h5"
LEARNING_RATE = 0.0001  # required for minimizing a loss function
EPOCHS = 40  # how many times the algorithm passes through all the dataset
BATCH_SIZE = 32 # number of samples for each pass forward pass
NUM_KEYWORDS = 10 # all the keywords in our data set


def load_dataset(data_path):

    with open(data_path, "r") as f:
        data = json.load(f)

    # extract inputs and targets
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y


def get_data_splits(data_path, test_size=0.1, test_val=0.1):

    # load dataset
    X, y = load_dataset(data_path)

    # create test, validation and test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_val)


    # convert inputs from 2d to 3d arrays
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):

    # network architecture
    model = keras.Sequential()

    # 1st convolutional layer
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # 2nd convolutional layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # 3rd convolutional layer
    model.add(keras.layers.Conv2D(32, (2, 2), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    # flattening the output of last conv layer into a dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    # softmax classifier
    model.add(keras.layers.Dense(NUM_KEYWORDS, activation="softmax"))

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=error, metrics=["accuracy"])

    # print model overview
    model.summary()

    return model


def main():

    # clear session
    keras.backend.clear_session()

    # load train, validation and test data splits
    X_train, X_val, X_test, y_train, y_val, y_test = get_data_splits(DATA_PATH)

    # build Convolutional Neural Network
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape, LEARNING_RATE)

    # training
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

    # test the model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {test_error}. Test accuracy: {test_accuracy}")

    # save model
    model.save(SAVED_MODEL_PATH)


if __name__ == '__main__':
    main()