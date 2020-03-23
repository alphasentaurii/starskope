# starskøpe

> "We have lingered long enough on the shores of the cosmic ocean. We are ready at last to set sail for the stars." ― Carl Sagan
=================================
# `S†∆R∫Køp∑ *[ªï]`
### `λ * φ * ƒ * ø`
Exoplanet Hunting and Stellar Classification: 
Unsupervised Machine Learning with Deep Boltzman Machines
=================================

*Capstone Project for the Flatiron School Datascience Bootcamp*

>>> NAME: Ru Kein
>>> DATE: 3-X-2020
>>> INSTRUCTOR: James Irving PhD
>>> COHORT: Full-time

----------------------
# Abstract 
----------------------
## `[ MISSION BRIEF]`
- Build a Machine Learning classification algorithm for detecting stars outside our solar system likely to be hosting exoplanets. 

## `[GOAL]`
- The underlying priority of this mission is, of course, to find *earth-like* exoplanet host-stars - the so-called "Hunt for Planet X".

## `[ANALYSIS]`
- Timeseries Analysis: observed measurements of flux values over time.
- Target: changes in flux indicating evidence of a threshold crossing event 
- Modeling: if star is likely to be hosting at least one exoplanet, aka `Exoplanet-Hosts`.

## `[DATASETS]`

K2 Light Curves, Target Pixel Files (Timeseries, Photometry, Imaging)
Kepler Light Curves, Target Pixel Files, Full-Frame Images (Time Series Analysis)
TESS Light Curves, Full Frame Images (Computer Vision)

## `[MODEL]`
* Heavy reliance on Deep Autoencoders and Data Augmentation*
* Pre-training / pre-learning approach to autoencoding
- Harmonic Frequency Analysis and Processing -> Fast Fourier Transformation ((FFT))
- Identification of harmonic coordinates across linear and nonlinear relationships 
- Four Vectors (based on Lorentz Transformation and Vector anaysis)

## `[CONTENTS]`
Part I: Supervised Learning - For developing our initial model, we will start with a time-series analysis of photometry-based light curves from the Kepler (K2) data sets, starting with Campaigns 1-4, tuning parameters as necessary before incorporating additional campaigns. 

Part II: Unsupervised Learning - Incorporate TESS direct image (FITS) data to build an unsupervised machine learning model that relies on computer vision to A) make accurate classifications with the lowest possible false probabilities. 

The practical outcome of the project seeks to classify known and unknown candidates as potential host-stars of earth-like planets using light curves and direct imaging data. 

To test and measure validity of the model's accuracy, we will calculate the most appropriate scores for data sets with highly imbalanced classes (since a simple Accuracy Score would in and of itself be an inaccurate assessment of the truth):
	* F1 Score (Harmonic mean of precision and recall)
	* Jaccard Index and Distance
	* Fowlkes-Mallows Index

## `[FUTURE DEVELOPMENTS - Starskøpe v2]`
1 - Build a a highly interactive, visually appealing and game-like simulator (via React frontend possibly) for astronomers, physicists, and citizen-scientists  to conduct research and contribute findings/notes in a virtual reality like setting - IE an AI telescope in cyberspace with real-time data from actual telescopes.

----------------

### BACKGROUND

> Question #1: 
Is there a certain mathematical system of analysis that can be universally applied to the identification and classification of stars (and other astronomical objects of interest)?

> Question #2:
Can this mathematical system (which we will use to develop into a statistical model) be derived from fundamental laws of physics and nature relating to harmonic frequencies and the concept of symmetry? 

> Additional questions:
- What are the [most important] underlying coordinates of stellar parameters in terms of the classification as an exoplanet host-star?
- Can identification of the same underlying coordinates be used for other classification problems such as pulsars, asteroids, etc? 
- What are the features that matter, and which ones can we ignore?
- Can these coordinates be identified accurately and efficiently using a time-series analysis of light curve photometry data? 
- Can these be accurately and efficiently identified using computer vision analysis of direct imaging data? 

### Additional Notes
The algorithm's pre-learning stage encoding is dependent on coordinates from three key domains: 
	1. Fourier Analysis (Harmonic coefficients)
	2. Fibonacci Series (symmetry)
	3. Sophie Germain Primes
	4. Four-Vectors  (Lorentz Transformation)

---


# FOURIER COEFFICIENTS and HARMONIC FREQUENCY ANALYSIS

"Can we write down the recipe if we are given the cake?" -Richard Feynman

Objective; develop an unsupervised machine learning model that accurately identifies exoplanet
host stars - both confirmed and unconfirmed - while producing minimal false positives.

The model's pre-training phase will involve encoding an algorithm with key theories from mathematics, physics and statistics: 

>	Fourier Analysis : given f(t), calculate all coefficients of various harmonic terms.
	(Determine what amount of each harmonic is required)
	
	Assumptions: 
	1. Any periodic frequency "can be represented by a suitable combination of harmonics" (R. Feynman [1]).

	2. Any sinusoidal oscillation at the frequency `w` can be written as the sum of terms `cos(wt)` and `sin(wt)`

	3. Any function that is periodic with period T can be written mathematically as:
		f(t) = a0
			+ a1coswt + b1sinwt
			+ a2cos2wt + b2sin2wt
			+ a3cos3wt + b3sin3wt
		where w = 2pi/T and a's and b's are numerical constants which tell us how much of each component oscillation is present in oscillation f(t).
	
	>> Likewise, we derive from this teh Energy Theorem : total energy in a wave is sum of energies in all former components. The sum of squares of reciprocals of odd integers is pi^2/8, and:
	pi^2/8 = (1 + 1/2^4 + 1/3^4 + ...) = pi^4/90
	
> Fibonacci Series 
	- Geometric shapes whose coordinates that display symmetry   

> Sophie Primes
	- p` = 2p + 1

## PROCESS

K2 photometric light curve data via MAST api

### K2 Technical Specifications and Background Info
The Kepler space telescope was launched with one goal in mind: to find earth-like planets orbiting sun-like stars.

K2's Campaign 3 commenced the telescope's seventh month of operation on Nov 12. The campaign 3 field-of-view includes more than 16,000 target stars, as well as observations of a number of objects within our own solar system, including the dwarf planet (225088) 2007 OR10, the largest known body without a name in the solar system, and the planet Neptune and its moon Nereid.

Campaign 3 had a nominal duration of 80 days, but an actual duration of 69.2 days. The campaign ended earlier than expected because the on-board storage filled up faster than anticipated due to poorer than expected data compression. [SOURCE: https://keplerscience.arc.nasa.gov/k2-data-release-notes.html#k2-campaign-3]

Kepler and K2 had three data modes: long cadence (30 min) and short cadence (1 min) postage stamps (TPFs), and quarterly FFIs (30 min). 


What is a TCE?
A: TCE stands for Threshold Crossing Event and is identified by the Kepler pipeline. A Threshold Crossing Event (TCE) is a sequence of transit-like features in the flux time series of a given target that resembles the signature of a transiting planet to a sufficient degree that the target is passed on for further analysis. For more information, see the Kepler documentation list. The interactive TCE table is available here.


### challenges
- Managing a Mess of Data
- false positives
Two common false positives — grazing eclipsing binaries (left) and background eclipsing binaries (right) — can mimic the signal of a transiting planet. [NASA/Ames Research Center]

https://aasnova.org/2018/12/07/using-machine-learning-to-find-planets/

"Recent years have seen a boom in exoplanet research — in large part due to the enormous data sets produced by transiting exoplanet missions like Kepler and, now, TESS. But the >3,000 confirmed Kepler planets weren’t all just magically apparent in the data! Instead, the discovery of planets is the result of careful classification of transit-like signals amid a sea of false positives from things like stellar eclipses and instrumental noise."

- CONTAMINATION: https://filtergraph.com/8717312
Contamination ratio of all CTL stars


TESS:
This Candidate Target List (CTL v7.02) is a compilation of several catalogs, including 2MASS, Gaia DR1, UCAC-4 & 5, Tycho-2, APASS DR9 and others. The CTL is the current best effort to identify stars most suitable for transit detection with TESS. Stars are considered for the CTL if they are: 1) identified as RPMJ dwarfs with greater than 2-sigma confidence; and 2) meet one of the following temperature/magnitude criteria: (TESSmag < 12 and Teff >= 5500K) or (TESSmag < 13 and Teff < 5500K). Alternatively, a star is included in the CTL, regardless of the conditions above, if the star is a member of the bright star list (TESSmag < 6) or the specially curated cool dwarf, hot subdwarf, and known planet lists. Users who are interested only in the top 200K or 400K stars may use a filter on the priority of 0.0017 and 0.0011 respectively.
The full TIC & CTL will be available for download at MAST. The full machine-readable version of this CTL filtergraph portal is available as a comma-separated file at this link.
CTL v7.02 was prepared for delivery by the TESS Target Selection Working Group to the TESS Science Office June 17, 2018. The full documentation for TIC v7 can be found here. The data release notes for this version of the TIC/CTL can be found here. Portal users are urged to routinely check the release notes for updates. Use of this database is governed by the TESS Catalog Fair Use Policy.

# Datasets

`k2`

`tess` 

`high time resolution survey (HTRU)`

'hst` (hubble)

Transit Surveys
TESS Project Candidates
K2 Targets (Search)
K2 Candidates Table
K2 Names Table
CoRoT Astero-Seismology
SuperWASP Light Curves (Search)
KELT Light Curves (Search)
KELT Praesepe Light Curves
XO Light Curves
HATNet Light Curves
Cluster Light Curves
TrES Light Curves

# Dashboard
 http://docs.astropy.org/en/stable/generated/examples/index.html


 # Appendix
 NASA Exoplanet Archive
 https://exoplanetarchive.ipac.caltech.edu/docs/API_exoplanet_columns.html
 DOI 10.26133/NEA1


# License
 https://hakkeray.mit-license.org/