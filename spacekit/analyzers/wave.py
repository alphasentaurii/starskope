
### MAKE_SPECGRAM
# create function for generating and saving spectograph figs
@staticmethod
def make_specgram(signal, Fs=None, NFFT=None, noverlap=None, mode=None,
                  cmap=None, units=None, colorbar=False, 
                  save_for_ML=False, fname=None,num=None,**kwargs):
    import matplotlib
    import matplotlib.pyplot as plt
    if mode:
        mode=mode
    if cmap is None:
      cmap='binary'

    #PIX: plots only the pixelgrids -ideal for image classification
    if save_for_ML == True:
        # turn off everything except pixel grid
        fig, ax = plt.subplots(figsize=(10,10),frameon=False)
        fig, freqs, t, m = plt.specgram(signal, Fs=Fs, NFFT=NFFT, mode=mode,cmap=cmap);
        plt.axis(False)
        plt.show();

        if fname is not None:
            try:
                if num:
                    path=fname+num
                else:
                    path=fname
                plt.savefig(fname+num,**pil_kwargs)
            except:
                print('Something went wrong while saving the img file')

    else:
        fig, ax = plt.subplots(figsize=(13,11))
        fig, freqs, t, m = plt.specgram(signal, Fs=Fs, NFFT=NFFT, mode=mode,cmap=cmap);
        plt.colorbar();
        if units is None:
            units=['Wavelength (λ)','Frequency (ν)']
        plt.xlabel(units[0]);
        plt.ylabel(units[1]);
        if num:
            title=f'Spectogram_{num}'
        else:
            title='Spectogram'
        plt.title(title)
        plt.show();

    return fig, freqs, t, m

