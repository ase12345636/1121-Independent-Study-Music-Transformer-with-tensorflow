import tensorflow as tf
from dataset import train1,train2,vaild1,vaild2
from transformer import Transformer
from optimizer import CustomSchedule
from loss import masked_loss,masked_accuracy


num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=388,
    target_vocab_size=388,
    dropout_rate=dropout_rate)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

transformer.fit((train1),
                epochs=1,
                validation_data=[vaild1,vaild2])

transformer.save_weights("transformer.h5")
