# # http://docs.lightkurve.org/


# Finding periodic signals
# The lightkurve.periodogram module provides classes to help find periodic signals in light curves.

# Periodogram(frequency, power[, nyquist, …])

# Generic class to represent a power spectrum (frequency vs power data).

# LombScarglePeriodogram(*args, **kwargs)

# Subclass of Periodogram representing a power spectrum generated using the Lomb Scargle method.

# BoxLeastSquaresPeriodogram(*args, **kwargs)

# Subclass of Periodogram representing a power spectrum generated using the Box Least Squares (BLS) method.



import lightkurve as lk

pixels = lk.search_targetpixelfile("Kepler-10").download()
pixels.plot()

lightcurve = pixels.to_lightcurve()
lightcurve.plot()

exoplanet = lightcurve.flatten().fold(period=0.838)
exoplanet.plot()



