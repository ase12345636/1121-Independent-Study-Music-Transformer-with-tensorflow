import processor

# Encode midi file
encoded = processor.encode_midi('Piano-e-Compitition-data.mid')
print(encoded)


# Create new midi file according to the encoding
decided = processor.decode_midi(encoded, file_path='Test-midi.mid')


# Show the contains of midi file
ins = processor.pretty_midi.PrettyMIDI('Piano-e-Compitition-data.mid')
for i in ins.instruments:
    print(i.control_changes)
    print(i.notes)
