import wave
import numpy

def open_wave(path):
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
    t = numpy.fromstring(w.readframes(w.getnframes()), formatsample[samplewidth]).astype(float) / scalesample[samplewidth] - deltasample[samplewidth]
    samplefrequency = w.getframerate()
    return t, float(samplefrequency)



def save_wave(array, path, samplefrequency = None, samplewidth = 4):
    """Convert numpy 1D float array to WAVE file"""
    if not samplewidth in [1, 2, 4]:
        raise Exception("Samplewidth must be 1, 2 or 4, got " + str(samplewidth) + ".")
    if samplefrequency == None:
        raise Exception("Samplefrequency was not defined.")
    deltasample = [0, 1, 0, 0, 0]
    scalesample = [0, 127.5, 2 ** 15 - 1, 0, 2 ** 31 - 1]
    formatsample = ['', 'u1', '<i2', '', '<i4']
    array[array < -1] = -1
    array[array > 1] = 1
    w = wave.open(path, 'wb')
    w.setnchannels(1)
    w.setframerate(samplefrequency)
    w.setsampwidth(samplewidth)
    u = ((array + deltasample[samplewidth]) * scalesample[samplewidth]).astype(formatsample[samplewidth])
    w.writeframes(u.tostring())
    w.close()

    
    
def play_wave(path):
    """Play wave file"""
    print(path)
    import platform
    if platform.system() == 'Windows':
        import winsound
        winsound.PlaySound(path, 0)
    elif platform.system() == 'Darwin':
        import subprocess
        subprocess.call(["afplay", path])
    else:
        import subprocess
        subprocess.call(["play", path])
