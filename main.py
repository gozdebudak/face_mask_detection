import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, ZeroPadding2D, BatchNormalization
from tensorflow.keras import optimizers


def main():
    DATA_DIR = "../data"
    IMG_SIZE = 100
    CATEGORIES = ["with_mask", "without_mask"]
    data = []
    creating_data(DATA_DIR, CATEGORIES, data, IMG_SIZE)
    random.shuffle(data)

    X = []
    y = []

    for features, label in data:
        X.append(features)
        y.append(label)

    cnn(X, y, IMG_SIZE)


def creating_data(DATADIR, CATEGORIES, data, IMG_SIZE):
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in (sorted(os.listdir(path))):
            try:
                img_array = cv2.imread(os.path.join(path, img))  # getting images from files
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  #
                data.append([new_array, class_num])  # creating training data
            except Exception as e:
               pass  # Passing


def cnn(X, y, IMG_SIZE, num):
    # print(len(X))
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    # print(X.shape)
    y = np.array(y)
    y = tf.keras.utils.to_categorical(y, num)
    # print(y.shape)
    cnn_model = Sequential()

    cnn_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=X.shape[1:]))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))

    cnn_model.add(Flatten())

    cnn_model.add(Dense(64, activation='relu'))
    cnn_model.add(Dropout(0.3))
    cnn_model.add(Dense(32, activation='relu'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Dense(num, activation='softmax'))
    # cnn_model.summary()

    # sgd = optimizers.SGD(learning_rate=0.01)
    # rmsprop = optimizers.RMSprop(learning_rate=0.01)
    adam = optimizers.Adam(learning_rate=0.01)
    # adadelta = optimizers.Adadelta()

    cnn_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    cnn_model.fit(X, y, epochs=15, batch_size=100, validation_split=0.3)
    # score = cnn_model.evaluate(X, y, verbose=0)
    # print('Test loss:', '{:.4f}'.format(score[0]))
    # print('Test accuracy:', '{:.4f}'.format(score[1]))


main()
