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

        midi_sequence_1st_half = []
        temp = list(tf.cast(x, dtype=tf.int32).numpy().reshape(-1))
        for i in temp:
            if temp[i] != 389 or 390 or 0:
                midi_sequence_1st_half.append(temp[i])

        midi_sequence_2nd_half = []

        output_tensor = np.array([389]).reshape(1, 1)
        output_tensor = tf.convert_to_tensor(output_tensor, dtype=tf.int32)
        output_tensor = tf.cast(output_tensor, dtype=tf.float32)

        for i in tf.range(max_length):
            predictions = self.transformer(
                [x, output_tensor], training=False)

            # Select the last token from the `seq_len` dimension.
            # Shape `(batch_size, 1, vocab_size)`.
            predictions = predictions[:, -1:, :]

            predicted_id = tf.argmax(predictions, axis=-1)

            # Concatenate the `predicted_id` to the output which is given to the
            # decoder as its input.

            if predicted_id == 0:
                output_tensor = tf.concat(
                    [output_tensor, tf.cast(predicted_id, dtype=tf.float32)], 1)

            elif predicted_id == 390:
                break

            else:
                midi_sequence_2nd_half.extend(list(
                    tf.cast(predicted_id, dtype=tf.int32).numpy().reshape(-1)))
                output_tensor = tf.concat(
                    [output_tensor, tf.cast(predicted_id, dtype=tf.float32)], 1)

        midi_sequence_1st_half.extend(midi_sequence_2nd_half)

        processor.decode_midi(
            midi_sequence_1st_half, file_path='Valid-midi_complete.mid')
        processor.decode_midi(
            midi_sequence_2nd_half, file_path='Valid-midi_output_from_model.mid')
