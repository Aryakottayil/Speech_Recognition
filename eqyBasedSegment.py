
import numpy as np
import matplotlib.pyplot as plt
import librosa as lb

filePath = r'C:\Users\HP\Downloads\should.wav'

# to retain the native sampling rate
y, sr = lb.load(filePath,sr=None)
frate = 100
winSize = int(np.ceil(30e-3*sr)) # in samples
hopLength = int(np.ceil(len(y)/frate))

# frame the signal
sigFrames = lb.util.frame(y,frame_length=winSize,hop_length=hopLength)
# compute energy
sigSTE = np.sum(np.square(sigFrames),axis=0)
#plt.plot(sigSTE)

meanEgy = np.mean(sigSTE)
# get logical indices
x = sigSTE>meanEgy
# speech egy values
speechEgy = sigSTE[x]
# nonspeech egy vales
nonSpeechEgy = sigSTE[~x]

# indices where egy greater than threshold
speechIndices = np.where(sigSTE>meanEgy)
nonSpeechIndices = np.where(sigSTE<=meanEgy)

print('Num speech frames: ',str(speechIndices[0].shape[0]))
print('Num nonspeech frames: ',str(nonSpeechIndices[0].shape[0]))

speechTimes = lb.frames_to_time(speechIndices,sr=sr,hop_length=hopLength)
nonSpeechTimes = lb.frames_to_time(nonSpeechIndices,sr=sr,hop_length=hopLength)

