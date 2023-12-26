import os
import random
import numpy as np
import tensorflow as tf

from midi_processor import midi_Tokenization

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


MAX_TOKENS = 502


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


# handling training data
x = []
y = []

for i in range(1, 2):
    path = os.path.dirname(os.path.abspath(__file__))
    path += "/Dataset/train/train (" + str(i) + ").midi"
    encoded = midi_Tokenization(path)

    start = random.randrange(0, len(encoded)-1001,
                             1) if len(encoded) > 1001 else 0
    encoded = encoded[start:]

    a = encoded[:500]
    a.insert(0, 389)
    a.append(390)
    a += [0]*(MAX_TOKENS-len(a))
    x += a

    b = encoded[500:1001]
    b.insert(0, 389)
    b.append(390)
    b += [0]*(MAX_TOKENS+1-len(b))
    y += b

train_ds = tf.data.Dataset.from_tensor_slices(
    (np.array(x).reshape(-1, MAX_TOKENS), np.array(y).reshape(-1, MAX_TOKENS+1)))


train_ds = (train_ds
            .shuffle(200)
            .batch(5)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE))


# handling validation data
path = os.path.dirname(os.path.abspath(__file__))
path += "/Dataset/vaild/vaild (1).midi"
encoded = midi_Tokenization(path)

x_valid = encoded[:500]
x_valid.insert(0, 389)
x_valid.append(390)

y_valid = encoded[500:1001]
y_valid.insert(0, 389)
y_valid.append(390)


valid_ds = tf.data.Dataset.from_tensor_slices(
    (np.array(x_valid).reshape(1, -1), np.array(y_valid).reshape(1, -1)))

valid_ds = (valid_ds
            .shuffle(200)
            .batch(1)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE))


# teat:

# for (x, y_inputs), y_labels in train_ds.take(1):
#     break

# print(x.shape)
# print(y_inputs.shape)
# print(y_labels.shape)

# print(x[0][:])
# print(y_inputs[0][:])
# print(y_labels[0][:])
