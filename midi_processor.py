import processor
import numpy as np
import tensorflow as tf


def midi_Tokenization(pm):
    encoded = processor.encode_midi(pm)

    return encoded
