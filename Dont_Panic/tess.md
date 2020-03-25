
All Exoplanets	4126
Confirmed Planets with Kepler Light Curves for Stellar Host	2356
Confirmed Planets Discovered by Kepler	2347
Kepler Project Candidates Yet To Be Confirmed	2420
Confirmed Planets with K2 Light Curves for Stellar Host	430
Confirmed Planets Discovered by K2	397
K2 Candidates Yet To Be Confirmed	891
Confirmed Planets Discovered by TESS 1	41
TESS Project Candidates Integrated into Archive (2020-02-19 04:30:01) 2	1660
Current date TESS Project Candidates at ExoFOP	
1737
TESS Project Candidates Yet To Be Confirmed 3	1105
1 Confirmed Planets Discovered by TESS refers to the number planets that have been published in the refereed astronomical literature.

2 TESS Project Candidates refers to the total number of transit-like events that appear to be astrophysical in origin, including false positives as identified by the TESS Project.

3 TESS Project Candidates Yet To Be Confirmed refers to the number of TESS Project Candidates that have not yet been dispositioned as a Confirmed Planet or False Positive.






- TESS 
classification using deep neural networks 
centroids and stellar parameters, as well as 1D light-curve
all sectors of available data
compare to previously known candidates (other surveys)
compare to vetted for accuracy (only 1000 so far?)
compare to results from Ansdell's team's predictions and model accuracy (validate or invalidate)

Plotting and EDA
 http://docs.astropy.org/en/stable/generated/examples/index.html


a star’s rotation, which can drive flares, slows as the star ages.

stars were sorted by galactic height and then counted what fraction of the stars exhibited flares at each height.

younger stars tend to lie closer to the midplane.


1. started wtih straightforward local and global views of the light curves
2. distinguish background eclipsing binary stars from planet transit signals
	> include data with each light curve:
		* how the line centroids: the pixel positions of the center of light — moved over time
		* known stellar parameters with the light curves.

	> Scoring: “Exonet” (Ansdell et al 2018) can classify a Kepler data set with:
		- 97.5% accuracy
		- 98% average precision. 
		
		That means that 97.5% of its classifications — exoplanet or false-positive — are correct, 
		and an average of 98% of transits classified as planets are true planets. 
		Recall (top; the fraction of true planets recovered) and precision (bottom; 
		the fraction of classifications that are correct) of the Exonet model, as a function of MES, a measure of the signal-to-noise of candidate transits. [Ansdell et al. 2018]


SECTORS 1-4: 
* 1000 stars vetted

* Every ~27.1 day “sector” monitors the light of tens of thousands of stars 

* which are then compiled into 1D “light curves”, 
* detrended for instrumental systematics
* searched for signals similar to transiting planets.

19 vetters completed the initial vetting stage of 1000 candidates or threshold crossing events (TCEs), with each candidate viewed by at least two vetters.

first attempts at classification using neural networks have tended to use exclusively the light curve [Shallue & Vanderburg (2018)] 
 
Ansdell et al. (2018) for FDL: modified the 1D light-curve-only neural network approach to candidate classification to include both centroids and stellar parameters, subsequently improving the precision of classification.
* first time deep learning has been performed for TESS.
* Had to simulate data for much the training validation bc TESS was only launched in 2018 and hadn't sent any data yet


SOURCE: https://www.aanda.org/articles/aa/full_html/2020/01/aa35345-19/aa35345-19.html


1 Introduction
In the next two years, the NASA Transiting Exoplanet Survey Satellite (TESS) mission (Ricker et al. 2014) is likely to more than double the number of currently known exoplanets (Sullivan et al. 2015; Huang et al. 2018a; Barclay et al. 2018). It will do this by observing 90% of the sky for up to one year and monitoring millions of stars with precise-enough photometry to detect the transits of extrasolar planets across their stars (e.g. Huang et al. 2018b; Vanderspek et al. 2019; Wang et al. 2019). 

* Every ~27.1 day “sector” monitors the light of tens of thousands of stars which are then compiled into 1D “light curves”, detrended for instrumental systematics, and searched for signals similar to transiting planets. However, those signals with exoplanetary origin are dwarfed by signals from false positives – those from artificial noise sources (e.g. systematic errors not removed by detrending), or from astrophysical false positives such as binary stars and variables. The best way to classify exoplanetary signals is therefore a key open question.


# VETTING:
Answers until now include human vetting, both by teams of experts (Crossfield et al. 2018) or members of the public (Fischer et al. 2012), vetting using:
* classical tree diagrams of specific diagnostics (Mullally et al. 2016), 
* ensemble learning methods such as random forests (McCauliff et al. 2015; Armstrong et al. 2018)
* deep learning techniques such as neural networks (Shallue & Vanderburg 2018; Schanche et al. 2018; Ansdell et al. 2018). 

The current process of vetting TESS candidates involves a high degree of human input. In Crossfield et al. (2018), 19 vetters completed the initial vetting stage of 1000 candidates or threshold crossing events (TCEs), with each candidate viewed by at least two vetters. However each TESS campaign has so far produced more than 1000 TCEs, and a simple extrapolation suggests as many as 500 human work hours may be required per month to select the best TESS candidates.

The first attempts at classification using neural networks have tended to use exclusively the light curve (e.g. Shallue & Vanderburg 2018; Zucker & Giryes 2018). In Ansdell et al. (2018), we modified the 1D light-curve-only neural network approach to candidate classification of Shallue & Vanderburg (2018) to include both centroids and stellar parameters, subsequently improving the precision of classification. In this paper we show results on adapting those models to both simulated and real TESS data, the first time deep learning has been performed for TESS.






-------------
FUTURE MISSIONS

1.  “SOFIA/HAWC+ Traces the Magnetic Fields in NGC 1068,” 
Enrique Lopez-Rodriguez et al 2020 ApJ 888 66.doi:10.3847/1538-4357/ab5849

2. The upcoming joint ESA/NASA Solar Orbiter mission, launching in February 2020, is just what we need: 
this spacecraft will eventually orbit the Sun at an inclination of 25° and above, allowing us a more definitive 
look at the Sun’s poles.

3. By combining the multi-year data from Evryscope with the much shorter but more precise observations from TESS, 
the authors of today’s paper attempt to get a handle on the frequency of a wide range of flare types from a large 
sample of stars.

4. a team of scientists at Predictive Science Inc. in San Diego have proposed a solution: 
what if this “missing” flux takes the form of concentrated bundles of magnetic flux at the 
solar poles that we just can’t see from our angle?

Earth’s position near the plane of the Sun’s equator makes it difficult for us to observe much 
of what’s happening at its poles. It’s entirely possible that there’s extra magnetic flux at the
poles of the Sun — where we can’t resolve it with ground-based observatories or Earth-based spacecraft — 
that our models are missing. By adding this flux into our models, maybe we’ll be able to reproduce the 
interplanetary magnetic field measured at Earth.

explore a series of global models of the Sun, reconstructing the coronal magnetic field and measuring the 
resulting magnetic flux at 1 AU. In their models, the authors add extra bundles of open magnetic field lines
at the poles of the Sun, and they then test whether this addition creates any other changes that would conflict 
with current solar observations.

5. The authors combine ground and space-based measurements for 4,068 K and M stars that were observed by 
both telescopes. Using their custom-made Auto-EFLS flare detection pipeline, the authors marked 284 of these 
objects as flare stars, with 575 flare events detected in total. The most energetic flare detection, produced 
by a small M dwarf, temporarily increased the brightness of the star by a factor of nearly 100. 
This superflare event released hundreds of years’ worth of the Sun’s energy output in a tiny fraction of that time! 
stars and their height above the disk of our galaxy.



 It’s there that our global solar models have a problem: they often underpredict the strength of the 
 interplanetary magnetic field at a distance of 1 AU (i.e., at Earth, where we can easily measure it) by 
 a factor of two or more.

So where’s the missing magnetic flux?
https://filtergraph.com/tess_ctl
https://aasnova.org/2020/02/11/tess-reveals-hd118203-b-transits-after-13-years/
https://aasnova.org/2020/02/18/the-tess-missions-first-earth-like-planet-found-in-an-interesting-trio/

https://aasnova.org/2020/02/04/why-are-there-so-many-sub-neptune-exoplanets/
https://aasnova.org/2018/12/07/using-machine-learning-to-find-planets/


