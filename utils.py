import fasttext
import json
import csv
from string import punctuation as p


def preprocess_sentence(sentence):

    sentence = sentence.lower()

    return sentence.translate(str.maketrans('', '', p))


def it_sentences(sentence_data):
    for line in sentence_data:
        example = json.loads(line)
        yield example['sentence2']


def it_labels(label_data):
    label_data_reader = csv.DictReader(label_data)
    for example in label_data_reader:
        yield example['gold_label']


def dev_accuracy(model):

    with open("snli_1.0_dev_filtered.jsonl","r") as data_sentences, \
         open("snli_1.0_dev_gold_labels.csv","r") as data_labels:

        correct_count = 0
        total_predictions = 0

        for label, sentence in zip(it_labels(data_labels),
                                   it_sentences(data_sentences)):

                predicted_label_aux = model.predict(preprocess_sentence(sentence))[0][0]
                predicted_label = predicted_label_aux.split('__label__')[1]

                total_predictions += 1
                correct_count += (predicted_label == label)

    return correct_count/total_predictions
