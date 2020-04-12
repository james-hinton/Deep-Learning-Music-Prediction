import guitarpro
import collections
from music21 import *
from convertToText import *
from os import listdir
import math
import rnntwo
from utils import *
from midiutil.MidiFile import MIDIFile
import pickle



def main(): 
    # Finds guitar pro files located in tabs
    mydir = './tabs'
    files = listdir(mydir)

    # Specify amount of epochs
    n_epochs = 36

    # For each note e.g. C, E ,G
    noteList = []

    # Converts each gp file into a list
    for gpfile in files:
        try:
            print('Found key for ', gpfile)
            # Parse guitar pro file
            curl = guitarpro.parse(mydir + '/'+ gpfile)
            songKey = curl.key.name
            # If default value (C Major) then don't add to list
            if str(songKey) == "CMajor":
                # Remove if default value
                os.remove(mydir + '/' +gpfile)
                print('Removed -', gpfile,'- Key not found')
            else:
                # Send to to get converted
                convertToList(noteList, curl)
        except:
            # Error handling
            print('FILE ERROR WITH ', gpfile)
            pass

    # Stores notes for use by validation
    with open('models/notes', 'wb') as filepath:
        pickle.dump(noteList, filepath)

    # Send to RNN
    model= rnntwo.build_model(noteList, n_epochs)

if __name__== '__main__':
    main()
