import pickle
import numpy
import sys
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.models import load_model
import keras
import guitarpro
from convertToText import *
from midiutil.MidiFile import MIDIFile

def main():
    # Load model
    model = keras.models.load_model('./models/modelInterval.hdf5')
    # Get the last note and input notes
    inputNotes, lastNote = readInput()
    # Load original notes for dict later
    with open('models/notes', 'rb') as filepath:
        originalNotes = pickle.load(filepath)

    network_input, pitchnames, n_vocab = processOriginal(originalNotes, inputNotes)
    prediction_output = predict(network_input, model, pitchnames, n_vocab)
    createSong(prediction_output, lastNote)

# Read the input file
def readInput():
    noteList = []
    myFile = './input/input.gp5'
    curl = guitarpro.parse(myFile)

    convertToList(noteList, curl)
    # Get last note
    lastNote = noteList[-1]
    return noteList, lastNote


def processOriginal(notes, inputNotes):
    # Arrange data into same as original build
    '''
    This retrieves help from Sigurður Skúli's guide:
    https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
    Website was accessed between July to September 2019.

    '''
    n_vocab = len(set(notes))
    sequence_length = 16
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    inputNotes = inputNotes[0]

    network_input = []
    # convert to rolling list and into dictionary key
    for i in range(0, len(inputNotes) - sequence_length, 1):
        sequence_in = inputNotes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])

    numpy.set_printoptions(threshold=sys.maxsize)
    return network_input, pitchnames, n_vocab


def predict(network_input, model, pitchnames, n_vocab):
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    prediction_output = []
    # Pattern last input
    pattern = network_input[-1]
    # Create a song with 30 notes
    for noteIndex in range(30):
        # Reshape data
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)
        # Predict
        prediction = model.predict(prediction_input, verbose=0)
        # Find data result from dict
        index = numpy.argmax(prediction)
        result = int_to_note[index]
        number = index + result
        prediction_output.append(result)

        pattern.append(number)
        pattern = pattern[1:len(pattern)]
    print(prediction_output)
    return prediction_output

def createSong(notes, lastNote):
        # Create MIDI file
        MIDI = MIDIFile(1)
        # Load MIDI data
        track = 0
        time=0
        MIDI.addTrackName(track, time, "sample")
        MIDI.addTempo(track,time,120)
        channel = 0
        # Add notes to MIDI
        addNote(MIDI, track, time, channel, notes, lastNote)

        with open("output.mid", 'wb') as outf:
            # Save MIDI file
            MIDI.writeFile(outf)
            print('Midi output created in directory folder')

def addNote(MIDI, track, time, channel, notes, lastNote):
    # Start at 60 middle
    note = 60
    for i in notes:
        note = note + i
        time += 1
        pitch = note
        volume = 100
        duration = 1
        MIDI.addNote(track, channel, pitch, time, duration, volume)


if __name__== '__main__':
    main()
