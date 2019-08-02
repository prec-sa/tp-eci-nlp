import fasttext
from utils import dev_accuracy
import gc

train_file = 'data_val.csv'


def generate_single_score(params):

    model = fasttext.train_supervised(train_file, thread = 1, **params)
    acc = dev_accuracy(model)

    gc.collect()

    return (params, acc)
