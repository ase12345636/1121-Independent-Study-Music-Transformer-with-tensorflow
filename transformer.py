import tensorflow as tf
from encoder import Encoder
from decoder import Decoder

# from dataset import train_ds


class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, num_linear_layer, d_model, num_heads, dff,
                 input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()

        self.num_linear_layer = num_linear_layer

        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=input_vocab_size,
                               dropout_rate=dropout_rate)

        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               vocab_size=target_vocab_size,
                               dropout_rate=dropout_rate)

        self.linear_layer = [
            tf.keras.layers.Dense(
                target_vocab_size)
            for _ in range(num_linear_layer)]

        self.final_layer = tf.keras.layers.Dense(
            target_vocab_size, activation="softmax")

    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        context, x = inputs

        context = self.encoder(context)  # (batch_size, context_len, d_model)

        x = self.decoder(x, context)  # (batch_size, target_len, d_model)

        # Final linear layer output.
        # (batch_size, target_len, target_vocab_size)
        for i in range(self.num_linear_layer):
            x = self.linear_layer[i](x)

        logits = self.final_layer(x)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            # b/250038731
            del logits._keras_mask
        except AttributeError:
            pass

        # Return the final output and the attention weights.
        return logits


<<<<<<< HEAD
# teat:


=======
>>>>>>> 92946ace48e59a87c04fca725f3f5571845d2237
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
# print(output)
