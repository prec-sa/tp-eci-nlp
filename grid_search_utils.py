import fasttext
from utils import dev_accuracy
import gc
from config import TRAIN_THREADS

train_file = 'data_val.csv'


def generate_single_score(params):

    model = fasttext.train_supervised(train_file, thread = TRAIN_THREADS,
                                      **params)
    acc = dev_accuracy(model)

    gc.collect()

    return (params, acc)
