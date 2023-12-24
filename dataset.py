from midi_processor import midi_Tokenization
import os
import numpy as np
import processor
import numpy
import os
import tensorflow as tf
import os
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from pprint import pprint
from IPython.display import clear_output
import tensorflow_datasets as tfds
import midi_processor

# train1=[]
# train2=[]
# for i in range (1,3):
#     path=os.path.dirname(os.path.abspath(__file__))
#     path+="/Dataset/train/train ("+ str(i) +").midi"
#     encoded0, encoded1 = midi_Tokenization(path)
#     train1.append(encoded0)
#     train2.append(encoded1)
# # print(train1)
# # print(len(train2))
# print("Finsh training dataset")

# vaild1=[]
# vaild2=[]
# for i in range (1,2):
#     path=os.path.dirname(os.path.abspath(__file__))
#     path+="/Dataset/vaild/vaild ("+ str(i) +").midi"
#     encoded0, encoded1 = midi_Tokenization(path)
#     vaild1.append(encoded0)
#     vaild2.append(encoded1)


# print("Finsh vailding dataset")


MAX_TOKENS = 1002


def prepare_batch(x, y):
    x = x[:, :MAX_TOKENS]
    x = tf.convert_to_tensor(x, dtype=tf.int32)
    x = tf.cast(x, dtype=tf.float32)

    y = y[:, :(MAX_TOKENS+1)]
    y_inputs = tf.convert_to_tensor(
        y[:, :-1], dtype=tf.int32)
    y_inputs = tf.cast(y_inputs, dtype=tf.float32)

    y_labels = tf.convert_to_tensor(
        y[:, 1:], dtype=tf.int32)
    y_labels = tf.cast(y_labels, dtype=tf.float32)

    return (x, y_inputs), y_labels


x = []
y = []

for i in range(1, 100):
    path = os.path.dirname(os.path.abspath(__file__))
    path += "/Dataset/train/train (" + str(i) + ").midi"
    encoded = midi_Tokenization(path)

    a = encoded[:1000]
    a.insert(0, 389)
    a.append(390)
    x.append(a)

    b = encoded[1000:2001]
    b.insert(0, 389)
    b.append(390)
    y.append(b)


train_ds = tf.data.Dataset.from_tensor_slices(
    (np.array(x).reshape(1, -1), np.array(y).reshape(1, -1)))


train_ds = (train_ds
            .shuffle(200)
            .batch(5)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE))

# for (x, y_inputs), y_labels in train_ds.take(1):
#     break

# print(x.shape)
# print(y_inputs.shape)
# print(y_labels.shape)

# print(x[0][:])
# print(y_inputs[0][:])
# print(y_labels[0][:])

'''
for i in range (1,138):
    path=os.path.dirname(os.path.abspath(__file__))
    path+="/Dataset/vaild/vaild ("+ str(i) +").midi"
    encoded = midi_Tokenization(path)
    resource,target = np.array_split (encoded,2)

for i in range (1,178):
    path=os.path.dirname(os.path.abspath(__file__))
    path+="/Dataset/train/test ("+ str(i) +").midi"
    encoded = midi_Tokenization(path)
    print(encoded)
    resource,target = np.array_split (encoded,2)
'''
