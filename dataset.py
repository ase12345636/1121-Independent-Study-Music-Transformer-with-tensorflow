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

encoded = midi_processor.midi_Tokenization(
    "Piano-e-Compitition-data.mid")

x = encoded[:500]
x.insert(0, 389)
x.append(390)

y = encoded[500:1001]
y.insert(0, 389)
y.append(390)

# print(max(x))
# print(max(y))

train_ds = tf.data.Dataset.from_tensor_slices(
    (np.array(x).reshape(1, -1), np.array(y).reshape(1, -1)))


def prepare_batch(x, y):
    x = tf.convert_to_tensor(x, dtype=tf.int32)
    x = tf.cast(x, dtype=tf.float32)

    y_inputs = tf.convert_to_tensor(
        y[:, :-1], dtype=tf.int32)
    y_inputs = tf.cast(y_inputs, dtype=tf.float32)

    y_labels = tf.convert_to_tensor(
        y[:, 1:], dtype=tf.int32)
    y_labels = tf.cast(y_labels, dtype=tf.float32)

    return (x, y_inputs), y_labels


train_ds = (train_ds
            .shuffle(200)
            .batch(1)
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
