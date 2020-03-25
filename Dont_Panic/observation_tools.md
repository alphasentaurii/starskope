

Observation planning
Web TESS Viewing (WTV) tool
The WTV tool allows users to check whether a target potentially falls within the TESS field of view (FOV). In addition, WTV can be used to calculate the brightness of a target in the TESS bandpass.

TESS-Point
This is a High Precision TESS pointing tool. It will convert target coordinates given in Right Ascension and Declination to TESS detector pixel coordinates for the first 13 TESS observing sectors (Year 1) focused on the southern ecliptic plane. It can also query MAST to obtain detector pixel coordinates for a star by TIC ID only. It provides the target ecliptic coordinates, sector number, camera number, detector number, and pixel column and row. If there is no output, then the target is not visible to TESS.

See our proposal tools page for additional resources that aid in the preparation of GI proposals.

TESS data analysis
Lightkurve is a Python-based package developed by the Kepler/K2 Guest Observer (GO) Office for use by the community to work with Kepler and K2 data. The TESS GI Office has partnered with the Kepler/K2 GO Office to develop lightkurve for use with TESS data.

The data formats are similar for Kepler/K2 and TESS: target pixel files (TPF) and full frame images (FFIs). Kepler and K2 had three data modes: long cadence (30 min) and short cadence (1 min) postage stamps (TPFs), and quarterly FFIs (30 min). 

TESS has two data modes, short cadence (2 min) postage stamps and 30 min cadence FFIs. Note that many tools are under development, and some are more robust than others. The TESS GI Office plans to update this list as new tools, software, and tutorials become available. *If you have any tools you would like us to include, please contact us at tesshelp@bigbang.gsfc.nasa.gov.*





`Detrending and analysis`
PyKE suite	The PyKE tools developed for the Kepler mission. The git repository can be found here.

*astrobase*	Light curve tools: periodograms (BLS, Lomb-Scargle, analysis of variance), simple detrending (fit high order polynomials), light-curve math (phase-folding, binning). Also, a server for vetting. A tutorial can be found here.


cave	Crowded Aperture Variability Extraction.

EVEREST	EPIC Variability Extraction and Removal for Exoplanet Science Targets; Detrending of K2 light curves.

halophot	K2 Halo Photometry for very bright stars.

K2-CPM	K2 Causal Pixel Model.

k2phot	Routines for extracting lightcurves from K2 images.

k2photometry	Read, reduce and detrend K2 photometry and search for transiting planets.

K2Pipeline	Data reduction and detrending pipeline for K2 data in Matlab.

k2sc	K2 systematics correction via simultaneous modelling of stellar variability and jitter-dependent systematics using Gaussian Process regression.

k2scTess	TESS systematics correction via simultaneous modelling of stellar variability and jitter-dependent systematics using Gaussian Process regression.

*keplersmear`	Make light curves from Kepler and K2 collateral data.*

nutella	Point spreads for Kepler/K2 inference.

*OxKeplerSC	Kepler jump and systematics correction using Variational Bayes and shrinkage priors.*

PySysRem	Correct systematic effects in large sets of photometric light curves.

skope	Synthetic K2 objects for PLD experimentation.






`Full frame image analysis`
*DIA	Difference Imaging Analysis to extract a light curve from FFIs.*

eleanor	eleanor is an open-source python framework for downloading, analyzing, and visualizing data from the TESS Full Frame Images.

f3	Full Frame Fotometry from the Kepler Full Frame Images.

FFIorBUST	Make light curves from the Kepler Full Frame Images.

Filtergraph	This is the TESS full-frame-image (FFI) portal which hosts the data products from the pipeline of Oelkers & Stassun (2018).

*HOTPANTS	High Order Transform of PSF and Template Subtraction; Similar method, but improvement on ISIS image subtraction processing. Documentation for HOTPANTS can be found here.*

ISIS	Process CCD images using image subtraction.
kepcal	Self calibration using the Kepler FFIs.

Lightkurve	Extract light curves from FFIs, and package into TPFs.

SpyFFI	Tools for simulating TESS imaging at multiple cadences, including cartoon light curves + jitter + focus drifts, cosmic rays.

TESSCut	Create time series pixel cutouts from the TESS FFIs. Find out what sectors/cameras/detectors a target was observed in.





`Positional tools`
k2-pix	Overlay a sky survey image on a K2 target pixel stamp.

k2ephem	Check whether a Solar System body is (or was) observable by K2.
 
k2fov	Check whether targets are in K2 FOV.

tess-point	Provides the target ecliptic coordinates, TESS sector number, camera number, detector number, and pixel column and row.

*tvguide	A tool for determining whether stars and galaxies are observable by TESS.*

`Data handling`
k2-quality-control	Automated quality control of Kepler/K2 data products.

k2flix	Create quicklook movies from the pixel data observed by Kepler/K2/TESS.

k2mosaic	Mosaic Target Pixel Files (TPFs) obtained by Kepler/K2 into images and movies.

kadenza	Converts raw cadence target data from the Kepler space telescope into FITS files.

kepFGS	Tools to use the Kepler and K2 Fine Guidance Sensor data.

keputils	Basic module for interaction with KOI and 
Kepler-stellar tables.

kplr	Tools for working with Kepler data using Python.

k2plr	Fork of dfm/kplr with added K2 functionality.

SuperstampFITS	Create individual FITS files of K2 superstamp regions.

`Planet search, modeling, and vetting`
*batman	Fast transit light curve models in Python.*

DAVE	Discovery And Vetting of K2 Exoplanets.

k2ps	K2 planet search.

Kepler-FLTI	Kepler Prime Flux-Level Transit Injection.

kepler-robovetter	The Kepler prime robovetter.

KeplerPORTS	The Kepler pipeline.

ketu	A search for transiting planets in K2 data.

ktransit	A simple exoplanet transit modeling tool in Python.

koi-fpp	False positive probabilities for all KOIs.

lcps	A tool for pre-selecting light curves with possible transit signatures.

planetplanet	A general photodynamical code for exoplanet light curves.

pysyzygy	A fast and general planet transit (syzygy) code written in C and in Python.

PyTransit	Fast and easy transit light curve modeling using Python and Fortran.

terra	Transit detection code.

ttvfast-python	Python interface to the TTVFast library.

VESPA Calculating false positive probabilities for transit signals.

`Miscellaneous science tools`
animate_spots	Make frames for animated gifs/movies showing a rotating spotted star.

appaloosa	Python-based flare finding code for Kepler light curves.

*celerite-asteroseis	Transit fitting and basic time-domain asteroseismology using celerite and ktransit.*

decatur	Tidal synchronization of Kepler eclipsing binaries.

*EMAC	The NASA Goddard Space Flight Center Exoplanet Modeling and Analysis Center (EMAC) serves as a repository and integration platform for modeling and analysis resources focused on the study of exoplanet characteristics and environments.*

***
*FoFreeAST	Fourier-Free Asteroseismology: uses celerite to model granulation and oscillations of stars.*
***

isochrones	Pythonic stellar model grid access; easy MCMC fitting of stellar properties.

*isoclassify	Perform stellar classifications using isochrone grids.*

kepler_orrery	Make a Kepler orrery gif or movie of all the Kepler multi-planet systems.

ldtk	Python toolkit for calculating stellar limb darkening profiles.

limb darkening	Limb-darkening and gravity-darkening coefficients for TESS.

MulensModel	Microlensing Modelling package.

*PandExo	A community tool for transiting exoplanets with HST & JWST.*

pymacula	Python wrapper for Macula analytic starspot code.

*PyOrbit	General toolkit for modeling radial velocity data.*

*radvel	Simultaneously characterize the orbits of exoplanets and the noise induced by stellar activity.*