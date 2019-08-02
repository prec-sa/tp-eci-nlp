import fasttext
import json
from collections import Counter 
import matplotlib.pyplot as plt
import numpy as np
import csv
model=fasttext.load_model("entrenamiento1.bin")

data=open("snli_1.0_test_filtered.jsonl","r")

model_results = open("model_results_nuevos.csv",'w', newline='')
mywriter = csv.writer(model_results)

labels=[]

for line in data:
        data_aux = json.loads(line)
        cosa=model.predict(data_aux['sentence2'].lower().translate(str.maketrans('', '', string.punctuation)))[0][0]
        cosita = cosa.split('__label__')
        labels.append(cosita[1])
        sentence_id = data_aux['pairID']
        output = sentence_id, cosita[1]
        mywriter.writerow(output)
    

#plt.plot(collections.Counter(labels))
#plt.show()

    
print(len(labels))    

etiquetas, values = zip(*Counter(labels).items())

indexes = np.arange(len(etiquetas))
width = 1

plt.bar(indexes, values, width)
plt.xticks(indexes + width * 0.5, etiquetas)
plt.show()
