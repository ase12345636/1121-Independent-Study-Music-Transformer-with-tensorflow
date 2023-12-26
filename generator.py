import tensorflow as tf
import processor
import numpy as np
import processor

MAX_TOKENS = 502


class MidiGenerator(tf.Module):
    def __init__(self, transformer):
        self.transformer = transformer

    def __call__(self, valid_ds, max_length=MAX_TOKENS):

        for (x, y_inputs), y_labels in valid_ds.take(1):
            break

        output_tensor = np.array([389]).reshape(1, 1)
        output_tensor = tf.convert_to_tensor(output_tensor, dtype=tf.int32)
        output_tensor = tf.cast(output_tensor, dtype=tf.float32)

        for i in tf.range(max_length):
            predictions = self.transformer(
                [y_inputs, output_tensor], training=False)

            # Select the last token from the `seq_len` dimension.
            # Shape `(batch_size, 1, vocab_size)`.
            predictions = predictions[:, -1:, :]

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.

            if predicted_id == 0:
                continue

            elif predicted_id == 390:
                break

            else:
                output_tensor = tf.concat(
                    [output_tensor, tf.cast(predicted_id, dtype=tf.float32)], 1)

        midi_sequence = list(
            tf.cast(output_tensor, dtype=tf.int32).numpy().reshape(-1))

        x = list(tf.cast(x, dtype=tf.int32).numpy().reshape(-1))[1:-1]
        x.extend(midi_sequence)

        processor.decode_midi(x, file_path='Valid-midi_complete.mid')
        processor.decode_midi(
            midi_sequence, file_path='Valid-midi_output_from_model.mid')
