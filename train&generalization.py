import tensorflow as tf
import os

from dataset import train_ds, valid_ds
from transformer import Transformer
from optimizer import CustomSchedule
from loss import masked_loss, masked_accuracy
from generator import MidiGenerator

num_layers = 5
num_linear_layer = 3
d_model = 512
dff = 1024
vocab_size = 388+3
num_heads = 4
dropout_rate = 0.05

transformer = Transformer(
    num_layers=num_layers,
    num_linear_layer=num_linear_layer,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=vocab_size,
    target_vocab_size=vocab_size,
    dropout_rate=dropout_rate)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

checkpoint_path = "TrainingCheckpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=50)

# transformer.fit(train_ds,
#                 epochs=300,
#                 callbacks=[cp_callback])

transformer.fit(train_ds,
                epochs=100)

midigenerator = MidiGenerator(transformer)
midigenerator(valid_ds)
