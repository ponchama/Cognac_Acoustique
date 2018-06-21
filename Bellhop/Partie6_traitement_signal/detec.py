import wave
import numpy as np

def open_wave(path) :
    """Convert WAVE file to numpy 1D float array and sample frequency"""
    w = wave.open(path, 'rb')
    if w.getnchannels() != 1:
        raise Exception("Only mono WAVE files are supported.")
    samplewidth = w.getsampwidth()
    if not samplewidth in [1, 2, 4]:
        raise Exception("Only 8, 16 or 32 bits WAVE files are supported.")
    deltasample = [0, 1, 0, 0, 0]
    scalesample = [0, 127.5, 2 ** 15 - 1, 0, 2 ** 31 - 1]
    formatsample = ['', 'u1', '<i2', '', '<i4']
    t = np.fromstring(w.readframes(w.getnframes()), formatsample[samplewidth]).astype(float) / scalesample[samplewidth] - deltasample[samplewidth]
    samplefrequency = w.getframerate()
    return t, float(samplefrequency)



### Signal d'entrée
signal, freq = open_wave( 'input_S.wav' )
signal_fft = np.fft.fft(signal)

### Signal de sortie

#s_out = np.load ('received_S_noise.npy')
#save_wave(s_out, 'received_S_noise.wav', samplefrequency = 48000., samplewidth = 4)
signal_out, freq_out = open_wave( 'received_S_noise.wav' )
signal_out_fft = np.fft.fft(signal_out)

### Convolution
Convol = signal_fft*signal_out_fft
Ifft = np.fft.ifft(Convol)
Signal1 = np.abs(Ifft)

seuil = 20.   ## sur quels critères doit-on fixer le seuil de détection ? 
t = np.arange(0., signal.size/freq, 1/freq)
t0 = 1.
dt_final = t[np.where(Signal1 > seuil)][0] - 2*t0
print("Le temps de trajet calculé par convolution est de %.6fs." %dt_final )