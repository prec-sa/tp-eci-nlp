import fasttext
import json
from collections import Counter 
import matplotlib.pyplot as plt
import numpy as np
import csv

def it_sentences(sentence_data):
    for line in sentence_data:
        example = json.loads(line)
        yield example['sentence2']

def it_labels(label_data):
    label_data_reader = csv.DictReader(label_data)
    for example in label_data_reader:
        yield example['gold_label']

model=fasttext.load_model("model_filename_2.bin")
#model=fasttext.load_model("entrenamiento.bin")

data_sentences=open("snli_1.0_dev_filtered.jsonl","r")
data_labels=open("snli_1.0_dev_gold_labels.csv","r")

correct_count = 0
total_predictions = 0


for label, sentence in zip(it_labels(data_labels),it_sentences(data_sentences)):
        predicted_label_aux=model.predict(sentence)[0][0]
        predicted_label = predicted_label_aux.split('__label__')[1]
        total_predictions=total_predictions+1
        if(predicted_label == label):
        	correct_count=correct_count+1
        else:
        	pass
print('Las predicciones totales son '+str(total_predictions))
print('Las predicciones correctas son '+str(correct_count))
print('La accuracy de tu modelo es de '+str("{:2.3f}".format(correct_count/total_predictions)))
        
#plt.plot(collections.Counter(labels))
#plt.show()

