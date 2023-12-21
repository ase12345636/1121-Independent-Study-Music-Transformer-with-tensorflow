import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text
import numpy as np
from transformer import Transformer

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True,
                               as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']
model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(
    f'{model_name}.zip',
    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
    cache_dir='.', cache_subdir='', extract=True
)

tokenizers = tf.saved_model.load('ted_hrlr_translate_pt_en_converter')

MAX_TOKENS=128

def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
    pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

    return (pt, en_inputs), en_labels

BUFFER_SIZE = 20000
BATCH_SIZE = 16
def make_batches(ds):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(prepare_batch, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))

train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

for (pt, en), en_labels in train_batches.take(1):

    transformer = Transformer(
                                num_layers=num_layers,
                                d_model=d_model,
                                num_heads=num_heads,
                                dff=dff,
                                input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
                                target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
                                dropout_rate=dropout_rate)
    output = transformer((pt, en))
    print(en.shape)
    print(pt.shape)
    print(output.shape)
    attn_scores = transformer.decoder.dec_layers[-1].last_attn_scores
    print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)
    transformer.summary()


