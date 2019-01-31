import json
import numpy as np
import tensorflow.keras
import tensorflow.keras.preprocessing.text as kpt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import model_from_json

MAX_WORDS = 5000

CATEGORIES = {
    'Bathroom furniture suites': [1, 0, 0, 0, 0, 0, 0],
    'Paint': [0, 1, 0, 0, 0, 0, 0],
    'Doors & door furniture': [0, 0, 1, 0, 0, 0, 0],
    'Kitchen worktops': [0, 0, 0, 1, 0, 0, 0],
    'Garden buildings': [0, 0, 0, 0, 1, 0, 0],
    'Plumbing': [0, 0, 0, 0, 0, 1, 0],
    'Power tools': [0, 0, 0, 0, 0, 0, 1]}


def get_category_name(label):
    for category, label in CATEGORIES.items():
        if np.array_equal(label, label):
            return category


def convert_text_to_index_array(text, dictionary):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
        else:
            print("'%s' not in training corpus; ignoring." % (word))
    return wordIndices


def load_model(model_path):
    # read in your saved model structure
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # and create a model from that
    model = model_from_json(loaded_model_json)
    # and weight your nodes with your saved values
    model.load_weights('model.h5')
    return model


def load_dictionary(dictionary_path):
    with open(dictionary_path, 'r') as dictionary_file:
        dictionary = json.load(dictionary_file)
    return dictionary


def main(string_to_classify):
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    dictionary = load_dictionary('dictionary.json')
    model = load_model('model.json')
    input_array = convert_text_to_index_array(string_to_classify, dictionary)
    input_sequence = tokenizer.sequences_to_matrix(
        [input_array], mode='binary')

    pred = model.predict(input_sequence)

    print("Item: %s, %s category; %f%% confidence.." %
          (string_to_classify, get_category_name([np.argmax(pred)]), pred[0][np.argmax(pred)] * 100))

    return get_category_name([np.argmax(pred)])
