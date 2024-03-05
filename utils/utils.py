import random
import numpy as np

def batch(iterable, batch_size):
    """Yields lists by bat
    """
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        # print(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

#split the dataset into training and validation dataset
def split_train_val(dataset, val_percent=0.1):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}



