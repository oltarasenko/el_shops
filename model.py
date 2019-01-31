import time
from tensorflow.keras import backend as K
import ipdb
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Sequential
import json
from tensorflow import keras
import tensorflow.keras.preprocessing.text as kpt
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd
import tensorflow as tf

from random import shuffle

MAX_WORDS = 5000

CATEGORIES = {
    'Bathroom furniture suites': [1, 0, 0, 0, 0, 0, 0],
    'Paint': [0, 1, 0, 0, 0, 0, 0],
    'Doors & door furniture': [0, 0, 1, 0, 0, 0, 0],
    'Kitchen worktops': [0, 0, 0, 1, 0, 0, 0],
    'Garden buildings': [0, 0, 0, 0, 1, 0, 0],
    'Plumbing': [0, 0, 0, 0, 0, 1, 0],
    'Power tools': [0, 0, 0, 0, 0, 0, 1]}

training_data = "/Users/olegtarasenko/repos/el_shops/training_set.csv"
testing_data = "/Users/olegtarasenko/repos/el_shops/testing_set.csv"


def get_label(name):
    return CATEGORIES[name]


def get_category_name(searched_label):
    for category, label in CATEGORIES.items():    # for name, age in dictionary.iteritems():  (for Python 2.x)
        if np.array_equal(label, searched_label):
            return category


def build_words_tokenizer(data):
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(data)
    return tokenizer


# Take input data & convert it to format clear to NN
def data_to_indicies(data, tokenizer):
    allWordIndices = []
    for text in data:
        word = convert_text_to_index_array(tokenizer.word_index, text)
        allWordIndices.append(word)

    allWordIndices = np.asarray(allWordIndices)
    return tokenizer.sequences_to_matrix(allWordIndices, mode='binary')


# Take input data & convert it to format clear to NN
def test_data_to_indicies(data, tokenizer):
    words = kpt.text_to_word_sequence(data)
    allWordIndices = []

    for word in words:
        if word in tokenizer.word_index:
            allWordIndices.append(word)

    allWordIndices = np.asarray(allWordIndices)
    return tokenizer.sequences_to_matrix(allWordIndices, mode='binary')

# Takes


def test_data_to_indicies(data, tokenizer):
    allWordIndices = []
    for text in data:
        word = convert_text_to_index_array(tokenizer.word_index, text)
        allWordIndices.append(word)

    allWordIndices = np.asarray(allWordIndices)
    return tokenizer.sequences_to_matrix(allWordIndices, mode='binary')


def convert_text_to_index_array(dictionary, text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [dictionary[word] for word in kpt.text_to_word_sequence(text) if dictionary.has_key(word)]


def read_data(path, keys):
    data = pd.read_csv(path)
    # print data
    data_set = []
    for _index, row in data.iterrows():
        items = []
        for key in keys:
            items.append(row[key])
        data_set.append(items)

    return data_set


def make_model(data, labels):
    model = Sequential()
    model.add(Dense(512, input_shape=(MAX_WORDS,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(data, labels,
              batch_size=500,
              epochs=4,
              verbose=1,
              validation_split=0.1,
              shuffle=True)

    return model


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(
            set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


data_set = read_data(training_data, ['descr', 'category_extracted'])
text = np.asarray([data[0] for data in data_set])

training_labels = np.array([get_label(i[1]) for i in data_set])

tokenizer = build_words_tokenizer(text)

training_data = data_to_indicies(text, tokenizer)
model = make_model(training_data, training_labels)
model.summary()
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

tf.train.write_graph(frozen_graph, "", "categorization.pb", as_text=False)

data_set = read_data(testing_data, ['descr', 'category_extracted'])

testing_text = np.asarray([data[0] for data in data_set])
testing_labels = np.array([get_label(i[1]) for i in data_set])

# print map_data(testing_text, tokenizer)

print "[error] Evaluating the model"
time.sleep(5)

test_loss, test_acc = model.evaluate(
    data_to_indicies(testing_text, tokenizer), testing_labels)

print "Loss: ", test_loss, "Accuracy:", test_acc
# # ipdb.set_trace()
# item = testing_set[2][0]
# print "Testing set:", item

# item_indices = test_data_to_indicies(item, tokenizer)
# print item_indices
# pred = model.predict(item_indices)
# ipdb.set_trace()
# print("%s item; category: %s, %f%% confidence" %
#       (item, get_category_name(training_labels[np.argmax(pred)]), pred[0][np.argmax(pred)] * 100))
