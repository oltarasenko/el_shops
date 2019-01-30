import ipdb
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Sequential
import json
from tensorflow import keras
import tensorflow.keras.preprocessing.text as kpt
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd

from random import shuffle

CATEGORIES = {
    'shoes': [1, 0, 0, 0, 0, 0, 0],
    'heater': [0, 1, 0, 0, 0, 0, 0],
    'lights': [0, 0, 1, 0, 0, 0, 0],
    'other': [0, 0, 0, 1, 0, 0, 0],
    'camera': [0, 0, 0, 0, 1, 0, 0],
    'watch': [0, 0, 0, 0, 0, 1, 0],
    'cleaning': [0, 0, 0, 0, 0, 0, 1]}

training_data = "/Users/olegtarasenko/repos/el_shops/training_data/trainig_data_prepared.csv"
testing_data = "/Users/olegtarasenko/repos/el_shops/testing_data/homebase2.csv"


def get_label(name):
    return CATEGORIES[name]


def get_category_name(searched_label):
    for category, label in CATEGORIES.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
        if np.array_equal(label, searched_label):
            return category


def build_words_tokenizer(data):
    max_words = 3000
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data)
    return tokenizer


# Take input data & convert it to format clear to NN
def data_to_indicies(data, tokenizer):
    allWordIndices = []

    for text in data:
        word = convert_text_to_index_array(tokenizer.word_index, text)
        # print "[error]", word
        # if word in tokenizer.word_index:
        allWordIndices.append(word)

    allWordIndices = np.asarray(allWordIndices)
    return tokenizer.sequences_to_matrix(allWordIndices, mode='binary')


# Take input data & convert it to format clear to NN
def test_data_to_indicies(data, tokenizer):
    words = kpt.text_to_word_sequence(data)
    allWordIndices = []

    for word in words:
        # word = convert_text_to_index_array(tokenizer.word_index, text)
        # print "[error]", word
        if word in tokenizer.word_index:
            allWordIndices.append(word)

    allWordIndices = np.asarray(allWordIndices)
    return tokenizer.sequences_to_matrix(allWordIndices, mode='binary')


def convert_text_to_index_array(dictionary, text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]


def read_data(path, keys):
    data = pd.read_csv(path)
    # print data
    data_set = []
    for _index, row in data.iterrows():
        items = []
        for key in keys:
            items.append(row[key])
        data_set.append(items)

    # data = [(row['descr'], row['category'])
    #         for _index, row in data.iterrows()]
    return data_set


def prepare_data(path):

    # shuffle data in place... to make learning better
    shuffle(data)

    # Prepare train labels
    labels = np.asarray([item[1] for item in data])
    labels = np.array([get_label(label) for label in labels])

    # Prepare train data
    data = np.asarray([item[0] for item in data])
    # max_words = 3000

    # tokenizer = Tokenizer(num_words=max_words)
    # tokenizer.fit_on_texts(data)
    # dictionary = tokenizer.word_index

    # with open('dictionary.json', 'w') as dictionary_file:
    #     json.dump(dictionary, dictionary_file)

    # allWordIndices = []

    # for text in data:
    #     wordIndices = convert_text_to_index_array(dictionary, text)
    #     allWordIndices.append(wordIndices)

    # allWordIndices = np.asarray(allWordIndices)

    # data = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
    return data, labels


def make_model(data, labels):
    model = Sequential()
    model.add(Dense(512, input_shape=(3000,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(data, labels,
              batch_size=32,
              epochs=5,
              verbose=1,
              validation_split=0.1,
              shuffle=True)

    return model


data_set = read_data(training_data, ['descr', 'category'])
text = np.asarray([data[0] for data in data_set])

training_labels = np.array([get_label(i[1]) for i in data_set])

tokenizer = build_words_tokenizer(text)
training_data = data_to_indicies(text, tokenizer)
model = make_model(training_data, training_labels)
model.summary()


testing_set = read_data(testing_data, ['descr', ])
# ipdb.set_trace()
item = testing_set[2][0]
print "Testing set:", item

item_indices = test_data_to_indicies(item, tokenizer)
print item_indices
pred = model.predict(item_indices)

print("%s item; category: %s, %f%% confidence" %
      (item, get_category_name(training_labels[np.argmax(pred)]), pred[0][np.argmax(pred)] * 100))
