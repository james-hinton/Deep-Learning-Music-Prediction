import guitarpro
import math
import os
from transpose import *


# Adds data from gp file into list
def convertToList(noteList, song):
    songKey = song.key.name
    semitone = transposeBy(songKey)
    print(songKey, ' to be transposed to CMajor by', semitone, 'semitones')
    # Keyword list for non-inclusive songs
    a = ['bass', 'cal', 'drum', 'rhythm', 'piano', 'organ', 'string'
    , 'vox', 'voice', 'key', 'synth', 'sax', 'choir', 'voix',
    'orgue', 'chour', 'coros', 'violin', 'base', 'chant', 'clav']

    # Iterate to get individual data
    for track in song.tracks:
        cumulativeBeat = 0
        # Checks not percussion track
        if not track.isPercussionTrack:
            trackName = track.name.lower()
            # Not in keyword list
            if not any(x in trackName for x in a):
                for measure in track.measures:    
                    for voice in measure.voices:
                        for beat in voice.beats:
                            # Checks not a chord
                            if len(beat.notes) == 1:
                                for note in beat.notes:
                                    convertednote = process(track, measure, voice, beat, note, semitone)
                                    noteList.append(convertednote)
                            



# Converts from note real value to note
def valueToNote(value):
    note = value % 12
    octave = (value / 12)
    octave = math.floor(octave)
    if note == 0:
        return 'C' + str(octave)
    elif note == 1:
        return 'C#'  + str(octave)
    elif note == 2:
        return 'D'  + str(octave)
    elif note == 3:
        return 'D#'+ str(octave)
    elif note == 4:
        return 'E'+ str(octave)
    elif note == 5:
        return 'F'+ str(octave)
    elif note == 6:
        return 'F#'+ str(octave)
    elif note == 7:
        return 'G'+ str(octave)
    elif note == 8:
        return 'G#'+ str(octave)
    elif note == 9:
        return 'A'+ str(octave)
    elif note == 10:
        return 'A#'+ str(octave)
    elif note == 11:
        return 'B'+ str(octave)

