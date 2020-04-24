

# def fastFourier(signal, bins):
    # """
    # takes in array and rotates #bins to the left as a fourier transform
    # returns vector of length equal to input array
    # """
    # import numpy as np
    # # 
    # arr = np.asarray(signal)
    # fq = np.arange(signal.size/2+1, dtype=np.float)
    # phasor = exp(complex(0.0, (2.0*np.pi)) * fq * bins / float(signal.size))
    # phaser = np.exp(complex(0.0, (2.0*np.pi)) * fq * 1 / float(sig.size))
# 
    # ff = np.fft.irfft(phasor * np.fft.rfft(signal))
    # 
    # return ff
    # 
    # 
# 
