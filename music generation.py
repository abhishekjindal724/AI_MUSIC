from keras.models import load_model
import numpy as np
import pickle
from music21 import converter, instrument, note, chord, stream

notes = []

with open ("notes", "rb") as file:
    notes = pickle.load(file)

pitch_names = sorted(set(notes))

model = load_model("model.hdf5")
int_to_element = dict ((num, element) for num, element in enumerate (pitch_names))
element_to_int = dict ((element, num) for num, element in int_to_element.items())

sequence_length = 100

test_input = []

for i in range (len(notes) - sequence_length):
    seq_inp = notes[i:i+sequence_length]
    test_input.append([element_to_int[ch] for ch in seq_inp])

vocab_len = 359
start = np.random.randint(len(test_input)-1)
pattern = test_input[start]
final_prediction = []
print("Running the model...")

for note_index in range(200):
    pred_inp = np.reshape(pattern, (1, len(pattern), 1))
    inp = pred_inp/float(vocab_len)

    prediction = model.predict(inp, verbose=0)
    idx = np.argmax(prediction)
    result = int_to_element[idx]
    final_prediction.append(result)
    pattern = pattern[1:]
    pattern.append(idx)

print("Music generated!")
print("Creating MIDI file...")

offset = 0  #Time
final_notes = []

for pattern in final_prediction:
    if ('+' in pattern ) or pattern.isdigit():
        notes_in_chord = pattern.split('+')
        temp_notes = []

        for curr_note in notes_in_chord:
            new_note = note.Note(int(curr_note))
            new_note.storedInstrument = instrument.Piano()
            temp_notes.append(new_note)

        new_chord = chord.Chord(temp_notes)
        new_chord.offset = offset
        final_notes.append(new_chord)
    else:
        curr_note = note.Note(pattern)
        curr_note.offset = offset
        curr_note.storedInstrument = instrument.Piano()
        final_notes.append(curr_note)

    offset += 0.5
    MIDI_stream = stream.Stream(final_notes)
MIDI_stream.write('midi', fp = 'output.mid')
print("Music saved as \"output.mid\" in current directory")