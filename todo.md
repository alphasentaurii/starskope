

# TODO

FIRST: What is the best way to classify exoplanetary signals?
SECOND: Can it be done accurately and reliably using only one type of data? EG timeseries-flux or 1D lightcurve analysis?
THIRD: how much more accurate/reliable would our algorithm be if:
1) training data includes both "LEFT-BRAIN" and "RIGHT-BRAIN" modes of thinking;
2) training data includes information that requires analytical processing using both patterns of thought, 
but each input is a meshed layering into one projected image.

For now, we must first take care of the basics and begin working with the most barebones, minimal dataset we can find: the Kepler labeled Time Series data found on Kaggle. 

	* signals with exoplanetary origin are dwarfed by signals from false positives
	due to artificial noise sources (e.g. systematic errors not removed by detrending), 
	or from astrophysical false positives such as binary stars and variables. 

	* Verification process requires a human - the size and frequency of data
	now being collected for analysis is growing at an exponentially larger scale.

	* My objective for this project is to develop an unsupervised machine learning 
	algorithm capable of classifying stars as 'candidate/non-candidate' to the highest
	possible degree of accuracy compared to methods that have so far been attempted 
	(Ansdell et al. 2018, Shallue & Vanderburg 2018; Schanche et al. 2018).

	* My initial model will rely strictly on transit detection methods applied 
	to data with labeled targets, drawing from a combined collection of surveys
	including Hubble (HST), Kepler/K2, and TESS. 

	* The final model is an unsupervised Deep Boltzman Machine that incorporates
	the use of non-linear autoencoders as a pre-training technique. The DBM will 
	be trained to perform image classification (computer vision) for both 
	labeled and unlabeled direct image data from TESS.

RELATED WORK:

* classical tree diagrams of specific diagnostics (Mullally et al. 2016), 
* ensemble learning methods such as random forests (McCauliff et al. 2015; Armstrong et al. 2018)
* deep learning techniques such as neural networks (Shallue & Vanderburg 2018; Schanche et al. 2018 
>> used only 1D light-curve; Ansdell et al. 2018 -- used centroids and stellar parameters, as well as 1D light-curve but used Simulated data because none was available yet)

[ ] OBTAIN
>> 1. Identify datasets from overlapping surveys
>> 2. Map coordinates, dimensions, units, functions for analysis and conversions
>> 3. Determine normalization, regularization, and other pre-processing techniques to apply
>> 4. Run initial models (PCA, Random Forest, etc) on labeled data
>> 5. Pre-training with autoencoders
>> 6. DBMs with labeled and unlabeled data

A. K2
* 1. Exoplanet labeled timeseries from Winter∆
	- get JD time stamps for C3 in MAST
	e.g. KIC 205889250
	- extract subset df containing flux for 42 confirmed planets (host stars)
	>> MAST:Kepler Confirmed dataset > KIC id, Start and End time, RA DEC, etc
			>> TPF
			>> LightCurve
			>> DVT 
			
2014-11-15T15:06:05

* 2. K2 Campaign 3 complete data 
	- MAST API

# Matching Kaggle Labeled Datasets to MAST API dataset

Pointing
RA: 336.66534641439 degrees
Dec: -11.096663792177 degrees
Roll: -158.494818065985 degrees

Targets With Data Available at MAST
16,833 EPIC IDs in long cadence (LC).
216 EPIC IDs in short cadence (LC).
Several custom targets (see below)

Full Frame Images (FFI)
ktwo2014331202630-c03_ffi-cal.fits
ktwo2015008010551-c03_ffi-cal.fits

First cadence
Time: 2014-11-15 14:06:05.515 UTC
Long Cadence Number: 99599
Short Cadence Number: 2976430

Last cadence
Time: 2015-01-23 18:37:04.488 UTC
Long Cadence Number: 102984
Short Cadence Number: 3078009
Most Recent Processing Version
Data Release 26



B. HUBBLE
	- AWS/HST API

C. TESS
	- AWS/HST API
	- FFI (Full Frame Images) via TessCut()

FUTURE WORK: 
>> Further Extend application of the model beyond exoplanets toward other key areas of research including: Pulsar Classification, Asteroid detection, nebula and solar flare analysis, and especially black holes.
>> Note: although launch of the James Web Space Telescope has been delayed into 2021 due to the coronavirus epidemic, extending the capacity of the model to retrieve and ajn 


# Stellar / Planetary Properties
## Parameters to set as benchmarks for classification of new observations
- KEPLER 
https://github.com/hakkeray/KeplerAI

- TESS 
classification using deep neural networks 
centroids and stellar parameters, as well as 1D light-curve
all sectors of available data
compare to previously known candidates (other surveys)
compare to vetted for accuracy (only 1000 so far?)
compare to results from Ansdell's team's predictions and model accuracy (validate or invalidate)

Plotting and EDA
 http://docs.astropy.org/en/stable/generated/examples/index.html

