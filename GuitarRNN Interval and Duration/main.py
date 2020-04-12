import guitarpro
import collections
from music21 import *
from convertToText import *
from os import listdir
import os.path
from os import path
import math
import rnnInterval
import rnnDuration
from utils import *
from midiutil.MidiFile import MIDIFile
import pickle



def main(): 
    # Finds guitar pro files located in tabs
    mydir = './tabs'
    files = listdir(mydir)

    n_epochs = 180

    noteList = [] # For each note e.g. C, E ,G
    noteDurationList = []

    # Converts each gp file into a list
    for gpfile in files:
        try:
            curl = guitarpro.parse(mydir + '/'+ gpfile)
            convertToList(noteList, curl)
            durationToList(noteDurationList, curl)
        except:
            pass
        
    # Convert noteList to correct data format
    combinedNotes = []
    for notes in noteList:
        for note in notes:
            combinedNotes.append(note)

    # Saves notes
    with open('models/notes', 'wb') as filepath:
        pickle.dump(combinedNotes, filepath)

    # Saves duration
    with open('models/durations', 'wb') as filepath:
        pickle.dump(noteDurationList, filepath)

    # Run interval model if it doesnt exist
    if path.exists("./models/modelInterval.hdf5") == False:
        rnnInterval.train_network(combinedNotes, n_epochs)
        pass
    else:
        print("note model exists")
    # Run duration model if it doesnt exist
    if path.exists("./models/modelDuration.hdf5") == False:
        rnnDuration.train_network(noteDurationList, n_epochs)
    else:
        print("duration model exists")

if __name__== '__main__':
    main()
