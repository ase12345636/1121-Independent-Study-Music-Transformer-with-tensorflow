import midi_processor
import os
import numpy as np

# Encode train midi file
for i in range (1,963):
    path=os.path.dirname(os.path.abspath(__file__))
    path+="/Dataset/train/train ("+ str(i) +").midi"
    encoded = midi_processor.encode_midi(path)
    resource,target = np.array_split (encoded,2)

for i in range (1,138):
    path=os.path.dirname(os.path.abspath(__file__))
    path+="/Dataset/vaild/vaild ("+ str(i) +").midi"
    encoded = midi_processor.encode_midi(path)
    resource,target = np.array_split (encoded,2)

for i in range (1,178):
    path=os.path.dirname(os.path.abspath(__file__))
    path+="/Dataset/train/test ("+ str(i) +").midi"
    encoded = midi_processor.encode_midi(path)
    print(encoded)
    resource,target = np.array_split (encoded,2)

'''
# Create new midi file according to the encoding
decided = midi_processor.decode_midi(encoded, file_path='sampleout.mid')


# Show the contains of midi file
ins = midi_processor.pretty_midi.PrettyMIDI(path)
for i in ins.instruments:
    print(i.control_changes)
    print(i.notes)
'''