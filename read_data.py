#!/usr/bin/env python
import argparse
import json
import csv
from utils import preprocess_sentence, it_sentences, it_labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sentences')
    ap.add_argument('labels', nargs='?')

    args = ap.parse_args()

    sentence_data = open(args.sentences, 'r')
    csv_out=open("data_val.csv","w")
    mywriter=csv.writer(csv_out, delimiter = '\t', escapechar = '\\',
                        quoting = csv.QUOTE_NONE)



    if args.labels:

        with open(args.labels, 'r') as label_data:

            for sentence, label in zip(it_sentences(sentence_data),
                                       it_labels(label_data)):
                # Tenemos la oración en sentence con su categoría en label

                output = ["__label__" + label, preprocess_sentence(sentence)]
                mywriter.writerow(output)

        csv_out.close()

    else:
        for sentence in it_sentences(sentence_data):
        # Tenemos una oración en sentence
            #print(sentence)
            pass


main()
