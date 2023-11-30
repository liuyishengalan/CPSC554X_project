import scipy.io
import librosa
import numpy as np

def zero_crossing_rate(signal):
    crossings = np.where(np.diff(np.sign(signal)))[0]
    rate = len(crossings) / len(signal)
    return rate

do_mfcc = False
do_timedomain = False
do_bandwidth = True

do_rms  = True      # root mean square
do_zcr = True       # zero crossing rate


if do_mfcc:
    print("Calculating MFCC features...")
    mfcc_feature = list()
    for i in range(20000):
        num = i+1
        filename = 'dataset/Tube_seg_1/audio_' + str(num) + '.mat'
        mat_data = scipy.io.loadmat(filename)
        wave = mat_data['Pr_Audio'].reshape(-1)
        srate = mat_data['srate'] * mat_data['srate_mul']
        mfcc = librosa.feature.mfcc(y=wave, sr=srate, n_mfcc=13, n_fft=len(wave), hop_length=len(wave))
        mfcc_feature.append(mfcc)
    np.save('mfcc_feature.npy', mfcc_feature)

if do_timedomain:
    print("Calculating time domain features...")
    td_feature = list()
    for i in range(20000):
        feature_list = list()
        num = i+1
        filename = 'dataset/audio_' + str(num) + '.mat'
        mat_data = scipy.io.loadmat(filename)
        wave = mat_data['Pr_Audio'].reshape(-1)
        
        if do_rms:
            rms = np.sqrt(np.mean(wave**2))
            feature_list.append(rms)
        
        if do_zcr:
            zcr = zero_crossing_rate(wave)
            feature_list.append(zcr)
        feature_array = np.array(feature_list)
        td_feature.append(feature_array)
        
    np.save('td_feature.npy', td_feature)

if do_bandwidth:
    print("Calculating bandwidth features...")
    s_b_feature = list()
    for i in range(20000):
        num = i+1
        filename = 'dataset/audio_' + str(num) + '.mat'
        mat_data = scipy.io.loadmat(filename)
        wave = mat_data['Pr_Audio'].reshape(-1)
        srate = mat_data['srate'] * mat_data['srate_mul']
        s_b = librosa.feature.spectral_bandwidth(y=wave, sr=srate, n_fft=2048, hop_length=512)
        s_b_feature.append(s_b)
    np.save('bandwidth_feature.npy', s_b_feature)

print("All done!")
