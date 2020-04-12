import guitarpro as guitarpro

# Finds how many semitones to transpose by to get C Major
def transposeBy(songKey):
    if songKey == "BMajorFlat":
        return 2
    if songKey == "BMajor":
        return 1
    if songKey == "CMajor":
        return 0
    if songKey == "CMajorSharp":
        return -1
    if songKey == "DMajorFlat":
        return -1
    if songKey == "DMajor":
        return -2
    if songKey == "DMajorSharp":
        return -3
    if songKey == "EMajorFlat":
        return -3
    if songKey == "EMajor":
        return -4
    if songKey == "FMajor":
        return -5
    if songKey == "FMajorSharp":
        return 6
    if songKey == "GMajorFlat":
        return 6
    if songKey == "GMajor":
        return 5
    if songKey == "GMajorSharp":
        return 4
    if songKey == "AMajorFlat":
        return 4
    if songKey == "AMajor":
        return 3
    if songKey == "AMajorSharp":
        return 2



# Converts note by semitones
def process(track, measure, voice, beat, note, semitone):
        note = note.realValue
        note = int(note)
        note += semitone
        return note



# Converts whole song to C
# Not in use
def convertToC(song, semitone):
    for track in song.tracks:
    		if not track.isPercussionTrack:
    			for measure in track.measures:
    				for voice in measure.voices:
    					for beat in voice.beats:
    						for note in beat.notes:
    							note = process(track, measure, voice, beat, note, semitone)
    print(note)
    return note
							

