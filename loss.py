import tensorflow as tf

# from dataset import train_ds
# from transformer import Transformer


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)


# for (x, y_inputs), y_labels in train_ds.take(1):
#     break

# num_layers = 4
# num_linear_layer = 3
# d_model = 512
# dff = 2048
# num_heads = 4
# dropout_rate = 0.1

# transformer = Transformer(
#     num_layers=num_layers,
#     num_linear_layer=num_linear_layer,
#     d_model=d_model,
#     num_heads=num_heads,
#     dff=dff,
#     input_vocab_size=388+3,
#     target_vocab_size=388+3,
#     dropout_rate=dropout_rate)

# output = transformer((x, y_inputs))

# print(x.shape)
# print(y_inputs.shape)
# print(y_labels.shape)
# print(output.shape)

# loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
#     reduction='none')
# loss = loss_object(y_labels, output)
# print(loss)

# mask = y_labels != 0
# mask = tf.cast(mask, dtype=loss.dtype)
# loss *= mask
# print(loss)

# print(tf.reduce_sum(loss, axis=1))
