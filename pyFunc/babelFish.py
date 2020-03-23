# UNIT CONVERSIONS

# https://specviz.readthedocs.io/en/latest/installation.html
# Spectrum Statistics
# 11 statistic/analysis functions are calculated using the input spectrum or specified region of interest, 
# depending on what is selected in the left side bar. If a region of interest is selected, the statistic calculations 
# are updated when that region of interest is changed.

# Calculations are done using the following functions:

# Mean
astropy.units.Quantity.mean
# Median:
numpy.median
# Std Dev:
astropy.units.Quantity.std
# Centroid:
specutils.analysis.centroid()
# RMS:
numpy.sqrt(flux.dot(flux) / len(flux))
# SNR:
specutils.analysis.snr()
# FWHM:
specutils.analysis.fwhm()
# Eq Width:
specutils.analysis.equivalent_width()
# Max:
astropy.units.quantity.Quantity.max
# Min:
astropy.units.quantity.Quantity.min
# Count Total:
specutils.analysis.line_flux()


astropy.units.equivalencies.spectral()
# Returns a list of equivalence pairs that handle spectral wavelength, wave number, frequency, and energy equivalences.
# Allows conversions between wavelength units, wave number units, frequency units, and energy units as they relate to light.

# There are two types of wave number:
# >> spectroscopic - 1/𝜆 (per meter)
# >> angular - 2𝜋/𝜆 (radian per meter)

"""
####
FunctionTransform(func, fromsys, tosys[, …])
A coordinate transformation defined by a function that accepts a coordinate object 
and returns the transformed coordinate object.

FunctionTransformWithFiniteDifference(func, …)
A coordinate transformation that works like a FunctionTransform, but computes velocity shifts 
based on the finite-difference relative to one of the frame attributes.
####
FunctionTransform(func, fromsys, tosys[, …])
A coordinate transformation defined by a function that accepts a coordinate object 
and returns the transformed coordinate object.

FunctionTransformWithFiniteDifference(func, …)
A coordinate transformation that works like a Function
"""

#==========# COORDINATES 

SkyCoord(*args[, copy])
High-level object providing a flexible interface for celestial coordinate representation, manipulation, 
and transformation between systems.


SkyCoordInfo([bound])
Container for meta information like name, description, format.

SkyOffsetFrame(*args, **kwargs)
A frame which is relative to some specific position and oriented to match its frame.


# http://docs.astropy.org/en/stable/coordinates/index.html#module-astropy.coordinates



GCRS(*args[, copy, representation_type, …])
A coordinate or frame in the Geocentric Celestial Reference System (GCRS).

Galactic(*args[, copy, representation_type, …])
A coordinate or frame in the Galactic coordinate system.

GalacticLSR(*args[, copy, …])
A coordinate or frame in the Local Standard of Rest (LSR), axis-aligned to the Galactic frame.

Galactocentric(*args, **kwargs)

A coordinate or frame in the Galactocentric system.


LSR(*args[, copy, representation_type, …])
A coordinate or frame in the Local Standard of Rest (LSR).


PhysicsSphericalDifferential(d_phi, d_theta, d_r)
Differential(s) of 3D spherical coordinates using physics convention.

PhysicsSphericalRepresentation(phi, theta, r)
Representation of points in 3D spherical coordinates 
(using the physics convention of using phi and theta for azimuth and inclination from the pole).

PrecessedGeocentric(*args[, copy, …])
A coordinate frame defined in a similar manner as GCRS, but precessed to a requested (mean) equinox.

QuantityAttribute([default, …])
A frame attribute that is a quantity with specified units and shape (optionally).

RadialDifferential(*args, **kwargs)

Differential(s) of radial distances.

RadialRepresentation(distance[, …])
Representation of the distance of points from the origin.

RangeError
Raised when some part of an angle is out of its valid range.




#RepresentationMapping
This namedtuple is used with the frame_specific_representation_info attribute to tell frames 
what attribute names (and default units) to use for a particular representation.
####
FunctionTransform(func, fromsys, tosys[, …])
A coordinate transformation defined by a function that accepts a coordinate object 
and returns the transformed coordinate object.

FunctionTransformWithFiniteDifference(func, …)
A coordinate transformation that works like a FunctionTransform, but computes velocity shifts 
based on the finite-difference relative to one of the frame attributes.
####
FunctionTransform(func, fromsys, tosys[, …])
A coordinate transformation defined by a function that accepts a coordinate object 
and returns the transformed coordinate object.

FunctionTransformWithFiniteDifference(func, …)
A coordinate transformation that works like a Function









SphericalCosLatDifferential(d_lon_coslat, …)
Differential(s) of points in 3D spherical coordinates.

SphericalDifferential(d_lon, d_lat, d_distance)
Differential(s) of points in 3D spherical coordinates.

SphericalRepresentation(lon, lat, distance)
Representation of points in 3D spherical coordinates.

StaticMatrixTransform(matrix, fromsys, tosys)
A coordinate transformation defined as a 3 x 3 cartesian transformation matrix.

Supergalactic(*args[, copy, …])
Supergalactic Coordinates (see Lahav et al.

TimeAttribute([default, secondary_attribute])
Frame attribute descriptor for quantities that are Time objects.

TransformGraph()
A graph representing the paths between coordinate frames.


# LIBRARIES AND HELPER PACKAGES

# https://heasarc.gsfc.nasa.gov/cgi-bin/tess/webtess/wtv.py

# 2MASS Catalog	2MASS JHHMMSSSSsDDMMSSS [A]	2MASS J23072869+2108033

# K2	K2-N	K2-1
# Note: Entering K2 with no number resolves to KOI-2.
# Kepler Input Catalog (KIC)	KIC NNNNNNN	KIC 6922244
# Kepler Object of Interest (KOI)	KOI NNNNN.NN	KOI 100.01

#   TESS Input Catalog (TIC)	TIC NNNNNNNNNN, or
# TIC NNNNNNNNNN.NN	TIC 1
# TIC 12345
# TIC 377780790
# TIC 181804752.01
# TESS Object of Interest (TOI)	TOI NNNNN.NN, or
# TOI NNNNN
# TOI 123.01
# TOI 98.02
# TOI 3711


# † In the LHS Catalog, some stellar entries are designated by a lower-case letter (e.g. LHS 5358b, which represents a star, not a planet). To query for a planet in such a system, use LHS NNNNa b.

# In the instructions above:

# N = an integer digit (0-9)
# HH = an integer number of hours (00-23)
# MM = an integer number of minutes (00-59)
# m = fraction of a minute (0-9); e.g., MMm = 217 corresponds to 21.7 minutes
# SSSS or SSS = seconds plus fraction of second (0000-5999 or 000-599); e.g., SSSS = 2248 corresponds to 22.48 seconds
# DD = integer number of degrees (00-89)
# YYYY = integer number of years (2000-2099)
# A = a letter of the (English) alphabet (A-Z or a-z)
# ggg = The full Greek letter (e.g., alpha, beta) or the 3 English letter abbreviation of Greek letters (e.g., alf, bet). For the Greek letters mu (μ) and nu (ν), please repeat the last character (muu, nuu) to enforce a three-character abbreviation.
# s = "+" or "-" sign
# CON = Three-letter abbreviation of constellation name (e.g., TAU, AND). See our complete list of constellation abbreviations here.

#https://exoplanetarchive.ipac.caltech.edu/applications/Inventory/search.html


# This new file appends two additional columns. 
# The number in the first column is the minimum number of 
# sectors the target is observable for and the second is 
# the maximum.

# source: Mukai, K. & Barclay, T. 2017, tvguide: A tool for determining whether stars and galaxies are observable by TESS., v1.0.0, Zenodo, doi:10.5281/zenodo.823357
# https://github.com/tessgi/tvguide


#install via PIP
# $ pip install tvguide --upgrade
# or GIT:
# $ git clone https://github.com/tessgi/tvguide.git
# $ cd tvguide
# $ python setup.py install


import tvguide

tvguide.check_observable(150.00, -60.00)

tvguide.check_many(ra_array, dec_array)


# $ tvguide 219.9009 -60.8356

# Success! The target may be observable by TESS during Cycle 1.
# We can observe this source for:
#     maximum: 2 sectors
#     minimum: 0 sectors
#     median:  1 sectors
#     average: 1.16 sectors


# You can also run on a file with targets currently implemented is using RA and Dec
# $ head inputfilename.csv

# 150., -60.
# 10., -75.
# 51., 0.
# 88., +65

# $ tvguide-csv inputfilename.csv

# Writing example-file.csv-tvguide.csv.

# $ head example-file.csv-tvguide.csv

# 150.0000000000, -60.0000000000, 0, 2
# 10.0000000000, -75.0000000000, 1, 3
# 51.0000000000, 0.0000000000, 0, 1
# 88.0000000000, 65.0000000000, 0, 0




######## K2fov - findcampaigns 
# https://github.com/KeplerGO/K2fov


# pip install K2fov

The simplest thing to do is to have a CSV file with columns "RA_degrees, Dec_degrees, Kepmag". Do not use a header.

For example, create a file called mytargetlist.csv containing the following rows:

178.19284, 1.01924, 13.2
171.14213, 5.314616, 11.3


K2findCampaigns
If instead of checking the targets in a single campaign, you want to understand whether a target is visible in any past or future K2 Campaign, you can use a different tool called K2findCampaigns.

Example

For example, to verify whether J2000 coordinate (ra, dec) = (269.5, -28.5) degrees is visible at any point 
during the K2 mission, type:

$ K2findCampaigns 269.5 -28.5
Success! The target is on silicon during K2 campaigns [9].
Position in C9: channel 31, col 613, row 491.
You can also search by name. For example, to check whether T Tauri is visible, type:

$ K2findCampaigns-byname "T Tauri"
Success! T Tauri is on silicon during K2 campaigns [4].
Position in C4: channel 3, col 62, row 921.
Finally, you can check a list of targets (either using their coordinates or names),
 using K2findCampaigns-csv. For example:

$ K2findCampaigns-csv targets.csv
Writing targets.csv-K2findCampaigns.csv.

$ K2inMicrolensRegion --help
usage: K2inMicrolensRegion [-h] ra dec

Check if a celestial coordinate is inside the K2C9 microlensing superstamp.

positional arguments:
  ra          Right Ascension in decimal degrees (J2000).
  dec         Declination in decimal degrees (J2000).

optional arguments:
  -h, --help  show this help message and exit








  #####

  # https://github.com/stephtdouglas/k2-pix.git

