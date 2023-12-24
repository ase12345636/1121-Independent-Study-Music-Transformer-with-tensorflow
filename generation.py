import os
import tensorflow as tf
from dataset import train_ds
from transformer import Transformer
from optimizer import CustomSchedule
from loss import masked_loss, masked_accuracy

# num_layers = 4
# num_linear_layer = 3
# d_model = 512
# dff = 2048
# vocab_size = 388+3
# num_heads = 4
# dropout_rate = 0.1

# transformer = Transformer(
#     num_layers=num_layers,
#     num_linear_layer=num_linear_layer,
#     d_model=d_model,
#     num_heads=num_heads,
#     dff=dff,
#     input_vocab_size=vocab_size,
#     target_vocab_size=vocab_size,
#     dropout_rate=dropout_rate)

# learning_rate = CustomSchedule(d_model)

# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                      epsilon=1e-9)

# transformer.compile(
#     loss=masked_loss,
#     optimizer=optimizer,
#     metrics=[masked_accuracy])

# transformer.train_on_batch(train_ds)

# transformer = tf.saved_model.load('transformer')
# transformer.
