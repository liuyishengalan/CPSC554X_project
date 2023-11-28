import scipy.io
import librosa
import numpy as np

filename = "dataset/audio_1.mat"
mat_data = scipy.io.loadmat(filename)
wave = mat_data['Pr_Audio'].reshape(-1)
print(type(wave))
srate = mat_data['srate'] * mat_data['srate_mul']
mfcc = librosa.feature.mfcc(y=wave, sr=srate, n_mfcc=13, n_fft=len(wave), hop_length=len(wave))
print(mfcc)

mfcc_feature = list()
for i in range(20000):
    num = i+1
    filename = 'dataset/Tube_seg_1/audio_' + str(num) + '.mat'
    wave = mat_data['Pr_Audio'].reshape(-1)
    srate = mat_data['srate'] * mat_data['srate_mul']
    mfcc = librosa.feature.mfcc(y=wave, sr=srate, n_mfcc=13, n_fft=len(wave), hop_length=len(wave))
    mfcc_feature.append(mfcc)

np.save('mfcc_feature.npy', mfcc_feature)