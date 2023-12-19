import processor
import numpy
import os
import tensorflow as tf

def midi_Tokenization(pm):
    pm = processor.pretty_midi.PrettyMIDI(pm)

    partitions = [0,pm.instruments[0].notes[-1].end/2,pm.instruments[0].notes[-1].end]

    for partition in range(len(partitions)-1):
        start_time = partitions[partition]
        end_time = partitions[partition + 1]

        new_midi= processor.pretty_midi.PrettyMIDI()
        for instr_num in range (len(pm.instruments)):
            instrument = (pm.instruments[instr_num])

            notes_velocity=[]
            notes_pitch=[]
            notes_start = []
            notes_end = []

        # 找出start_time之后的第一个音符编号记作note_num
            for start_note_num in range (len(instrument.notes)):
                note = instrument.notes[start_note_num]
                start = note.start
                if start >= start_time:
                    break

            for end_note_num in range (len(instrument.notes)):
                note = instrument.notes[end_note_num]
                end = note.end
                if end > end_time:
                    break
        #将原midi中，区间内的音符记下
            for k in range(start_note_num,end_note_num):
                note = instrument.notes[k]
                notes_pitch.append(note.pitch)
                notes_start.append(note.start)
                notes_end.append(note.end)
                notes_velocity.append(note.velocity)

            program = instrument.program
            is_drum = instrument.is_drum
            name = instrument.name
            inst = processor.pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
            new_midi.instruments.append(inst)

        # 粘到新midi里
            for i in range (end_note_num - start_note_num):
                inst.notes.append(processor.pretty_midi.Note(notes_velocity[i], notes_pitch[i], notes_start[i]-float(start_time), notes_end[i]-float(start_time)))

        new_midi.write(os.path.dirname(os.path.abspath(__file__))+"\Piano-e-Compitition-data_"+str(partition)+".mid")

    # Encode midi file
    encoded0 = numpy.array(processor.encode_midi(os.path.dirname(os.path.abspath(__file__))+"\Piano-e-Compitition-data_"+"0"+".mid"))

    encoded1 = numpy.array(processor.encode_midi(os.path.dirname(os.path.abspath(__file__))+"\Piano-e-Compitition-data_"+"1"+".mid"))

    return encoded0 , encoded1