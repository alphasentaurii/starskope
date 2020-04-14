
# `STARSKØPE`



**Building a Cyberoptic Neural Network Telescope for Astrophysical Object Classification**

> Flatiron School Capstone Project
* `Author: Ru Keïn`
* `Instructor: James Irving PhD`
* `Data Science Full-Time Program`
* `Blog post URL:` http://www.hakkeray.com/datascience/2020/03/22/planetX-hunter-classification-algorithms.html
* `Non-Technical Presentation`: Datascience-CAPSTONE-starskope.pdf

    Note: this project is divided into 3 notebooks:

    starskøpe I : Keras Neural Network Model (this notebook)
    starskøpe II: Computer Vision/Restricted Boltzmann Machines for Spectographs
    starskøpe III: CV/RBMs for Fourier Transformed Spectographs

# Mission Brief

## ABSTRACT

> "Mathematicians [...] are often led astray when 'studying' physics because they lose sight of the physics. 
They say: *'Look, these differential equations--the Maxwell equations--are all there is to electrodynamics; it is admitted by the physicists that there is nothing which is not contained in the equations. The equations are complicated, but after all they are only mathematical equations and if I understand them mathematically inside out, I will understand the physics inside out.'* Only it doesn't work that way. Mathematicians who study physics with that point of view--and there have been many of them--usually make little contribution to physics and, in fact, little to mathematics. They fail because the actual physical situations in the real world are so complicated that it is necessary to have a much broader understanding of the equations."
**-Richard Feynman, *The Feynman Lectures on Physics: Volume 2*, Chapter 2-1: "Differential Calculus of Vector Fields"**

---

**INTRODUCTION**
One of the reasons I quote Mr. Feynman above is because I set out to work on this project with only one year of high school physics under my belt. Despite loving the subject and even getting an A- in that one class, for some reason I did not continue pursuing physics while in school. I bought the Feynman lectures a few years back (on a whim? who does that?) and as soon as I began preparing for this project I felt intuitively that it would be somewhat ridiculous for me to build neural networks for classifying astrophysical data if I had no idea what the data meant beyond a surface level (i.e. Google), let alone on an intimate, perhaps even quantum scale. 

**BACKGROUND**
I'm intensely curious about why things work the way they do, and I'm not satisified by the answer unless I know the math behind it too. During the course of this Capstone project, I somehow managed to (that is, found it extremely necessary to) read almost all of Parts I and II of the Feynman lectures. I did that because I wanted to understand the physics, not just the math. The underlying question I am asking  -- and the ultimate argument I am proposing for astrophysics-related applications of machine learning in general -- is one that I believe Richard Feynman would agree with, were he around still today: *machine learning models for physics need to take physics into account, not simply the math.* After all, we train our own brains' neurons by learning physics before we go around making statements and predictions about the universe. Shouldn't we do the same for our algorithms?

**QUESTIONS**
For this project, the specific questions I am looking to answer are as follows: 

    1. Can a transiting exoplanet be detected strictly by analyzing the raw flux values of a given star? 
    2. What kind of normalization, de-noising, and/or unit conversion is necessary in order for the analysis to be accurate? 
    3. How much signal-to-noise ratio is too much? That is, if the classes are highly imbalanced, for instance only a few planets can be confirmed out of thousands of stars, does the imbalance make for an unreliable or inaccurate model? 
    4. How do we test and validate that?
  
To recap, the overall question is: how much actual physics do we need to know and understand as data scientists in order to make good (or preferably, great) data scientists for the world of space exploration (and thus the fields of astrophysics, astronomy, and aerospace)?

**DATASET**
To answer the above questions, I started the analysis with a labeled timeseries dataset from Kaggle posted by NASA several years ago. The reason I chose this particular dataset is because in terms of the type of information we typically need to know in order to solve a physics problem -- the primary one being UNITS, otherwise it's a math problem! -- this one is barren. The author who posted the dataset (`Winter Delta` or `W∆`) does however give us a few hints on how we *could* determine the units, and the dimensions, and a lot of other important physics-related information, if we do a little research. The biggest hint is that this dataset is from the K2 space telescope's Campaign 3 observations in which only 42 confirmed exoplanets are detected in a set of over 5,000 stars. Looking at the dataset on its own (before doing any digging), we are given little information about how long the time period covers, and we know do not know what the time intervals between flux values are. So far, this has not stopped any data scientists from attempting to tackle the classification model without gathering any additional information. 

**MODEL**
To answer the question, I first set out to build a model for the data as is, "sans-physics". The baseline model is a neural network using the Keras API in a sci-kit learn wrapper.  

**RESULTS**
I was able to identify with 99% accuracy the handful of stars (5) in the test dataset that have a confirmed exoplanet in their orbit. 

**CONCLUSION**
This baseline model is mathematically accurate, but it does not "understand physics". The conclusion we need to make about the model is whether or not this lack of physics embedded in the training process (or even pre-training process) is acceptable or not.

While it is possible to create a 99% accurate machine learning model for detecting exoplanets using the raw flux values, without any sense of the actual time intervals, and with a highly imbalanced data set (imbalanced meaning only a few positive examples in a sea of negatives) - it is unclear that we can "get away with" this in every case. Furthermore, it is unlikely that could feel completely sure that we aren't missing out on critical information - such as detecting the existence of an earth-like exoplanet transiting a star - if we don't use our understanding of physics to further de-noise, normalize, and scale the data before training the model (and possibly even embed this into a pre-training phase). As a case in point, if you read any of the space telescope handbooks, you will quickly learn just how complex the instruments that are producng this data are, and that the way their technology works, when and where in the sky they were pointing, as well as what actually happened during their missions, you'd know that should all probably be taken into account in your model! The K2 data in particular, for instance, has a unique issue that every so often its thrusters would fire to adjust/maintain its position in the sky, causing data at multiple points to be completely useless. 

*Why that matters...*
This type of noise cannot be removed without knowing what exact times the thrusters fired, as well as what times each of the observations of the dataset occurred. Even if we do manage to throw the bad data out, we are still stuck with the problem of not having any data for that time period, and once again might miss our potential planet's threshold crossing event! If we know where and when those missing pieces occur, we could use that to collect our missing data from another telescope like TESS, which has overlapping targets of observation. A model that can combine data from two different space telescopes, and be smart enough to know based on the telescope it came from how to handle the data, would make truly accurate predictions, and much more useful classifications. 

*What we can do about that...*
This is the type of model I will set out to build in my future work. This is what we would call a cyberoptic artificial telescope - one that can aggregate large datasets from multiple missions and give us a more accurate, more detailed picture of the stars and planets than what we have available to us in the limited view of a single picture from a single telescope at a single point in time. This is the vision for *STARSKØPE* which will come out of this project.

**RECOMMENDATIONS**
My recommendations are the following:

   1. Use datasets from the MAST website (via API) to incorporate other calculations of the star's properties as features to be used for classification algorithms. Furthermore, attempt other types of transformations and normalizations on the data before running the model - for instance, apply a Fourier transform.

   2. Combine data from multiple campaigns and perhaps even multiple telescopes (for instance, matching sky coordinates and time intervals between K2, Kepler, and TESS for a batch of stars that have overlapping observations - this would be critical for finding transit periods that are longer than the campaigns of a single telecope's observation period).

   3. Explore using computer vision on not only the Full Frame images we can collect from telescopes like TESS, but also on spectographs of the flux values themselves. The beauty of machine learning is our ability to rely on the computer to pick up very small nuances in differences that we ourselves cannot see with our own eyes. 
   
   4. Explore using autoencoded machine learning algorithms with Restricted Boltzmann Machines - this type of model has proven to be incredibly effective in the image analysis of handwriting as we've seen applied the MNIST dataset - let's find out if the same is true for images of stars, be they the Full Frame Images or spectographs.

**FUTURE WORK**
To continue this project, I'll take another approach for detecting exoplanets using computer vision to analyze images of spectographs of this same star flux data set. Please go to the notebook `[starskøpe-2]` to see how I use a Restricted Boltzmann Machines neural network model to classify stars as exoplanet hosts using spectograph images of the flux values to find transiting exoplanets. Following this, I will apply the same algorithm to spectographs of Fourier transformed data, as you will see in `[starskøpe-3]`. 

Additional future work following this project will be to develop my "cyberoptic artificial telescope" as a machine learning driven application that any astrophysicist can use to look at a single or collection of stars and have the model classify them according not only to exoplanet predictions, but also predict what type of star it is, and other key properties that would be of interest for astrophysical science applications.


# Obtain

Begin by importing libraries and code packages for basic analysis, as well as the kaggle dataset.


```python
# Import code packages and libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
%matplotlib inline
from matplotlib.colors import LogNorm

import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
 

font_dict={'family':'"Titillium Web", monospace','size':16}
mpl.rc('font',**font_dict)


#ignore pink warnings
import warnings
warnings.filterwarnings('ignore')
# Allow for large # columns
pd.set_option('display.max_columns', 0)
# pd.set_option('display.max_rows','')
```

Import dataset which has already been split into train and test sets, `exoTrain.csv.zip` and `exoTest.csv.zip` (I compressed them from their original csv format since the training set is > 240 MB so we'll to unzip them).


```python
import os, glob, sys

home = os.path.abspath(os.curdir)

os.listdir(home)
```




    ['Datascience-CAPSTONE-starskope.pdf',
     '.DS_Store',
     'LICENSE',
     'starskope-2.ipynb',
     '288_planetbleed1600.jpeg',
     'README.md',
     'starskope.ipynb',
     'starskope-2-colab.ipynb',
     '.gitignore',
     '_config.yml',
     '.ipynb_checkpoints',
     '.git',
     'DATA',
     'outputs']




```python
# %cd ../
%cd data
%ls
```

    /Users/hakkeray/CODE/CAPSTONE/starskope/DATA
    [1m[36m__MACOSX[m[m/         exoTest.csv       exoTrain.csv
    exoTableDraw.R    exoTest.csv.zip   exoTrain.csv.zip



```python
# uncomment below if you need to unzip the data files
# !unzip -q 'exoTrain.csv.zip'
# !unzip -q 'exoTest.csv.zip'
# %ls
```


```python
train = pd.read_csv('exoTrain.csv')
test = pd.read_csv('exoTest.csv')
```


```python
# cd backto home / root directory
%cd ../
```

    /Users/hakkeray/CODE/CAPSTONE/starskope


# Scrub

Initial inspection of data, reviewing the features, target (if any), datatypes, and checking for nulls.

LABEL is our target column, the remaining 3197 columns are the "features" which in this case make up the frequency of the signal from each star.

Each star's light frequency makes up a single row of data collected over the course of the campaign (#3), which in this case for K2 campaign 3 was a little over 60 days (campaigns are normally ~80 days but c3 ended early due to data storage capacity issues. 

If we crunch the numbers we'll see this means it's 29.4 minutes between each flux measurement, also known as the cadence. This also lines up with the information available in the K2 handbook.


```python
# comparing train and test datasets
display(train.head(), test.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LABEL</th>
      <th>FLUX.1</th>
      <th>FLUX.2</th>
      <th>FLUX.3</th>
      <th>FLUX.4</th>
      <th>FLUX.5</th>
      <th>FLUX.6</th>
      <th>FLUX.7</th>
      <th>FLUX.8</th>
      <th>FLUX.9</th>
      <th>FLUX.10</th>
      <th>FLUX.11</th>
      <th>FLUX.12</th>
      <th>FLUX.13</th>
      <th>FLUX.14</th>
      <th>FLUX.15</th>
      <th>FLUX.16</th>
      <th>FLUX.17</th>
      <th>FLUX.18</th>
      <th>FLUX.19</th>
      <th>FLUX.20</th>
      <th>FLUX.21</th>
      <th>FLUX.22</th>
      <th>FLUX.23</th>
      <th>FLUX.24</th>
      <th>FLUX.25</th>
      <th>FLUX.26</th>
      <th>FLUX.27</th>
      <th>FLUX.28</th>
      <th>FLUX.29</th>
      <th>FLUX.30</th>
      <th>FLUX.31</th>
      <th>FLUX.32</th>
      <th>FLUX.33</th>
      <th>FLUX.34</th>
      <th>FLUX.35</th>
      <th>FLUX.36</th>
      <th>FLUX.37</th>
      <th>FLUX.38</th>
      <th>FLUX.39</th>
      <th>FLUX.40</th>
      <th>FLUX.41</th>
      <th>FLUX.42</th>
      <th>FLUX.43</th>
      <th>FLUX.44</th>
      <th>FLUX.45</th>
      <th>FLUX.46</th>
      <th>FLUX.47</th>
      <th>FLUX.48</th>
      <th>...</th>
      <th>FLUX.3149</th>
      <th>FLUX.3150</th>
      <th>FLUX.3151</th>
      <th>FLUX.3152</th>
      <th>FLUX.3153</th>
      <th>FLUX.3154</th>
      <th>FLUX.3155</th>
      <th>FLUX.3156</th>
      <th>FLUX.3157</th>
      <th>FLUX.3158</th>
      <th>FLUX.3159</th>
      <th>FLUX.3160</th>
      <th>FLUX.3161</th>
      <th>FLUX.3162</th>
      <th>FLUX.3163</th>
      <th>FLUX.3164</th>
      <th>FLUX.3165</th>
      <th>FLUX.3166</th>
      <th>FLUX.3167</th>
      <th>FLUX.3168</th>
      <th>FLUX.3169</th>
      <th>FLUX.3170</th>
      <th>FLUX.3171</th>
      <th>FLUX.3172</th>
      <th>FLUX.3173</th>
      <th>FLUX.3174</th>
      <th>FLUX.3175</th>
      <th>FLUX.3176</th>
      <th>FLUX.3177</th>
      <th>FLUX.3178</th>
      <th>FLUX.3179</th>
      <th>FLUX.3180</th>
      <th>FLUX.3181</th>
      <th>FLUX.3182</th>
      <th>FLUX.3183</th>
      <th>FLUX.3184</th>
      <th>FLUX.3185</th>
      <th>FLUX.3186</th>
      <th>FLUX.3187</th>
      <th>FLUX.3188</th>
      <th>FLUX.3189</th>
      <th>FLUX.3190</th>
      <th>FLUX.3191</th>
      <th>FLUX.3192</th>
      <th>FLUX.3193</th>
      <th>FLUX.3194</th>
      <th>FLUX.3195</th>
      <th>FLUX.3196</th>
      <th>FLUX.3197</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2</td>
      <td>93.85</td>
      <td>83.81</td>
      <td>20.10</td>
      <td>-26.98</td>
      <td>-39.56</td>
      <td>-124.71</td>
      <td>-135.18</td>
      <td>-96.27</td>
      <td>-79.89</td>
      <td>-160.17</td>
      <td>-207.47</td>
      <td>-154.88</td>
      <td>-173.71</td>
      <td>-146.56</td>
      <td>-120.26</td>
      <td>-102.85</td>
      <td>-98.71</td>
      <td>-48.42</td>
      <td>-86.57</td>
      <td>-0.84</td>
      <td>-25.85</td>
      <td>-67.39</td>
      <td>-36.55</td>
      <td>-87.01</td>
      <td>-97.72</td>
      <td>-131.59</td>
      <td>-134.80</td>
      <td>-186.97</td>
      <td>-244.32</td>
      <td>-225.76</td>
      <td>-229.60</td>
      <td>-253.48</td>
      <td>-145.74</td>
      <td>-145.74</td>
      <td>30.47</td>
      <td>-173.39</td>
      <td>-187.56</td>
      <td>-192.88</td>
      <td>-182.76</td>
      <td>-195.99</td>
      <td>-208.31</td>
      <td>-103.22</td>
      <td>-193.85</td>
      <td>-187.64</td>
      <td>-92.25</td>
      <td>-119.25</td>
      <td>-87.50</td>
      <td>-1.86</td>
      <td>...</td>
      <td>2.15</td>
      <td>-6.04</td>
      <td>-58.44</td>
      <td>-29.64</td>
      <td>-90.71</td>
      <td>-90.71</td>
      <td>-265.25</td>
      <td>-367.84</td>
      <td>-317.51</td>
      <td>-167.69</td>
      <td>-56.86</td>
      <td>7.56</td>
      <td>37.40</td>
      <td>-81.13</td>
      <td>-20.10</td>
      <td>-30.34</td>
      <td>-320.48</td>
      <td>-320.48</td>
      <td>-287.72</td>
      <td>-351.25</td>
      <td>-70.07</td>
      <td>-194.34</td>
      <td>-106.47</td>
      <td>-14.80</td>
      <td>63.13</td>
      <td>130.03</td>
      <td>76.43</td>
      <td>131.90</td>
      <td>-193.16</td>
      <td>-193.16</td>
      <td>-89.26</td>
      <td>-17.56</td>
      <td>-17.31</td>
      <td>125.62</td>
      <td>68.87</td>
      <td>100.01</td>
      <td>-9.60</td>
      <td>-25.39</td>
      <td>-16.51</td>
      <td>-78.07</td>
      <td>-102.15</td>
      <td>-102.15</td>
      <td>25.13</td>
      <td>48.57</td>
      <td>92.54</td>
      <td>39.32</td>
      <td>61.42</td>
      <td>5.08</td>
      <td>-39.54</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>-38.88</td>
      <td>-33.83</td>
      <td>-58.54</td>
      <td>-40.09</td>
      <td>-79.31</td>
      <td>-72.81</td>
      <td>-86.55</td>
      <td>-85.33</td>
      <td>-83.97</td>
      <td>-73.38</td>
      <td>-86.51</td>
      <td>-74.97</td>
      <td>-73.15</td>
      <td>-86.13</td>
      <td>-76.57</td>
      <td>-61.27</td>
      <td>-37.23</td>
      <td>-48.53</td>
      <td>-30.96</td>
      <td>-8.14</td>
      <td>-5.54</td>
      <td>15.79</td>
      <td>45.71</td>
      <td>10.61</td>
      <td>40.66</td>
      <td>16.70</td>
      <td>15.18</td>
      <td>11.98</td>
      <td>-203.70</td>
      <td>19.13</td>
      <td>19.13</td>
      <td>19.13</td>
      <td>19.13</td>
      <td>19.13</td>
      <td>17.02</td>
      <td>-8.50</td>
      <td>-13.87</td>
      <td>-29.10</td>
      <td>-34.29</td>
      <td>-24.68</td>
      <td>-27.62</td>
      <td>-31.21</td>
      <td>-32.31</td>
      <td>-37.52</td>
      <td>-46.58</td>
      <td>-46.20</td>
      <td>-35.79</td>
      <td>-42.09</td>
      <td>...</td>
      <td>5.10</td>
      <td>17.57</td>
      <td>-16.46</td>
      <td>21.43</td>
      <td>-32.67</td>
      <td>-32.67</td>
      <td>-58.56</td>
      <td>-51.99</td>
      <td>-32.14</td>
      <td>-36.75</td>
      <td>-15.49</td>
      <td>-13.24</td>
      <td>20.46</td>
      <td>-1.47</td>
      <td>-0.40</td>
      <td>27.80</td>
      <td>-58.20</td>
      <td>-58.20</td>
      <td>-72.04</td>
      <td>-58.01</td>
      <td>-30.92</td>
      <td>-13.42</td>
      <td>-13.98</td>
      <td>-5.43</td>
      <td>8.71</td>
      <td>1.80</td>
      <td>36.59</td>
      <td>-9.80</td>
      <td>-19.53</td>
      <td>-19.53</td>
      <td>-24.32</td>
      <td>-23.88</td>
      <td>-33.07</td>
      <td>-9.03</td>
      <td>3.75</td>
      <td>11.61</td>
      <td>-12.66</td>
      <td>-5.69</td>
      <td>12.53</td>
      <td>-3.28</td>
      <td>-32.21</td>
      <td>-32.21</td>
      <td>-24.89</td>
      <td>-4.86</td>
      <td>0.76</td>
      <td>-11.70</td>
      <td>6.46</td>
      <td>16.00</td>
      <td>19.93</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>532.64</td>
      <td>535.92</td>
      <td>513.73</td>
      <td>496.92</td>
      <td>456.45</td>
      <td>466.00</td>
      <td>464.50</td>
      <td>486.39</td>
      <td>436.56</td>
      <td>484.39</td>
      <td>469.66</td>
      <td>462.30</td>
      <td>492.23</td>
      <td>441.20</td>
      <td>483.17</td>
      <td>481.28</td>
      <td>535.31</td>
      <td>554.34</td>
      <td>562.80</td>
      <td>540.14</td>
      <td>576.34</td>
      <td>551.67</td>
      <td>556.69</td>
      <td>550.86</td>
      <td>577.33</td>
      <td>562.08</td>
      <td>577.97</td>
      <td>530.67</td>
      <td>553.27</td>
      <td>538.33</td>
      <td>527.17</td>
      <td>532.50</td>
      <td>273.66</td>
      <td>273.66</td>
      <td>292.39</td>
      <td>298.44</td>
      <td>252.64</td>
      <td>233.58</td>
      <td>171.41</td>
      <td>224.02</td>
      <td>237.69</td>
      <td>251.53</td>
      <td>236.06</td>
      <td>212.31</td>
      <td>220.95</td>
      <td>249.08</td>
      <td>234.14</td>
      <td>259.02</td>
      <td>...</td>
      <td>-45.09</td>
      <td>-50.22</td>
      <td>-97.19</td>
      <td>-64.22</td>
      <td>-123.17</td>
      <td>-123.17</td>
      <td>-144.86</td>
      <td>-106.97</td>
      <td>-56.38</td>
      <td>-51.09</td>
      <td>-33.30</td>
      <td>-61.53</td>
      <td>-89.61</td>
      <td>-69.17</td>
      <td>-86.47</td>
      <td>-140.91</td>
      <td>-84.20</td>
      <td>-84.20</td>
      <td>-89.09</td>
      <td>-55.44</td>
      <td>-61.05</td>
      <td>-29.17</td>
      <td>-63.80</td>
      <td>-57.61</td>
      <td>2.70</td>
      <td>-31.25</td>
      <td>-47.09</td>
      <td>-6.53</td>
      <td>14.00</td>
      <td>14.00</td>
      <td>-25.05</td>
      <td>-34.98</td>
      <td>-32.08</td>
      <td>-17.06</td>
      <td>-27.77</td>
      <td>7.86</td>
      <td>-70.77</td>
      <td>-64.44</td>
      <td>-83.83</td>
      <td>-71.69</td>
      <td>13.31</td>
      <td>13.31</td>
      <td>-29.89</td>
      <td>-20.88</td>
      <td>5.06</td>
      <td>-11.80</td>
      <td>-28.91</td>
      <td>-70.02</td>
      <td>-96.67</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>326.52</td>
      <td>347.39</td>
      <td>302.35</td>
      <td>298.13</td>
      <td>317.74</td>
      <td>312.70</td>
      <td>322.33</td>
      <td>311.31</td>
      <td>312.42</td>
      <td>323.33</td>
      <td>311.14</td>
      <td>326.19</td>
      <td>313.11</td>
      <td>313.89</td>
      <td>317.96</td>
      <td>330.92</td>
      <td>341.10</td>
      <td>360.58</td>
      <td>370.29</td>
      <td>369.71</td>
      <td>339.00</td>
      <td>336.24</td>
      <td>319.31</td>
      <td>321.56</td>
      <td>308.02</td>
      <td>296.82</td>
      <td>279.34</td>
      <td>275.78</td>
      <td>289.67</td>
      <td>281.33</td>
      <td>285.37</td>
      <td>281.87</td>
      <td>88.75</td>
      <td>88.75</td>
      <td>67.71</td>
      <td>74.46</td>
      <td>69.34</td>
      <td>76.51</td>
      <td>80.26</td>
      <td>70.31</td>
      <td>63.67</td>
      <td>75.00</td>
      <td>70.73</td>
      <td>70.29</td>
      <td>95.44</td>
      <td>100.57</td>
      <td>114.93</td>
      <td>103.45</td>
      <td>...</td>
      <td>-18.86</td>
      <td>-11.27</td>
      <td>-19.92</td>
      <td>-1.99</td>
      <td>-13.49</td>
      <td>-13.49</td>
      <td>-27.74</td>
      <td>-30.46</td>
      <td>-32.40</td>
      <td>-2.75</td>
      <td>14.29</td>
      <td>-14.18</td>
      <td>-25.14</td>
      <td>-13.43</td>
      <td>-14.74</td>
      <td>2.24</td>
      <td>-31.07</td>
      <td>-31.07</td>
      <td>-50.27</td>
      <td>-39.22</td>
      <td>-51.33</td>
      <td>-18.53</td>
      <td>-1.99</td>
      <td>10.43</td>
      <td>-1.97</td>
      <td>-15.32</td>
      <td>-23.38</td>
      <td>-27.71</td>
      <td>-36.12</td>
      <td>-36.12</td>
      <td>-15.65</td>
      <td>6.63</td>
      <td>10.66</td>
      <td>-8.57</td>
      <td>-8.29</td>
      <td>-21.90</td>
      <td>-25.80</td>
      <td>-29.86</td>
      <td>7.42</td>
      <td>5.71</td>
      <td>-3.73</td>
      <td>-3.73</td>
      <td>30.05</td>
      <td>20.03</td>
      <td>-12.67</td>
      <td>-8.77</td>
      <td>-17.31</td>
      <td>-17.35</td>
      <td>13.98</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2</td>
      <td>-1107.21</td>
      <td>-1112.59</td>
      <td>-1118.95</td>
      <td>-1095.10</td>
      <td>-1057.55</td>
      <td>-1034.48</td>
      <td>-998.34</td>
      <td>-1022.71</td>
      <td>-989.57</td>
      <td>-970.88</td>
      <td>-933.30</td>
      <td>-889.49</td>
      <td>-888.66</td>
      <td>-853.95</td>
      <td>-800.91</td>
      <td>-754.48</td>
      <td>-717.24</td>
      <td>-649.34</td>
      <td>-605.71</td>
      <td>-575.62</td>
      <td>-526.37</td>
      <td>-490.12</td>
      <td>-458.73</td>
      <td>-447.76</td>
      <td>-419.54</td>
      <td>-410.76</td>
      <td>-404.10</td>
      <td>-425.38</td>
      <td>-397.29</td>
      <td>-412.73</td>
      <td>-446.49</td>
      <td>-413.46</td>
      <td>-1006.21</td>
      <td>-1006.21</td>
      <td>-973.29</td>
      <td>-986.01</td>
      <td>-975.88</td>
      <td>-982.20</td>
      <td>-953.73</td>
      <td>-964.35</td>
      <td>-956.60</td>
      <td>-911.57</td>
      <td>-885.15</td>
      <td>-859.38</td>
      <td>-806.16</td>
      <td>-752.20</td>
      <td>-792.40</td>
      <td>-703.91</td>
      <td>...</td>
      <td>-674.90</td>
      <td>-705.88</td>
      <td>-708.77</td>
      <td>-844.59</td>
      <td>-1023.12</td>
      <td>-1023.12</td>
      <td>-935.68</td>
      <td>-848.88</td>
      <td>-732.66</td>
      <td>-694.76</td>
      <td>-705.01</td>
      <td>-625.24</td>
      <td>-604.16</td>
      <td>-668.26</td>
      <td>-742.18</td>
      <td>-820.55</td>
      <td>-874.76</td>
      <td>-874.76</td>
      <td>-853.68</td>
      <td>-808.62</td>
      <td>-777.88</td>
      <td>-712.62</td>
      <td>-694.01</td>
      <td>-655.74</td>
      <td>-599.74</td>
      <td>-617.30</td>
      <td>-602.98</td>
      <td>-539.29</td>
      <td>-672.71</td>
      <td>-672.71</td>
      <td>-594.49</td>
      <td>-597.60</td>
      <td>-560.77</td>
      <td>-501.95</td>
      <td>-461.62</td>
      <td>-468.59</td>
      <td>-513.24</td>
      <td>-504.70</td>
      <td>-521.95</td>
      <td>-594.37</td>
      <td>-401.66</td>
      <td>-401.66</td>
      <td>-357.24</td>
      <td>-443.76</td>
      <td>-438.54</td>
      <td>-399.71</td>
      <td>-384.65</td>
      <td>-411.79</td>
      <td>-510.54</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3198 columns</p>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LABEL</th>
      <th>FLUX.1</th>
      <th>FLUX.2</th>
      <th>FLUX.3</th>
      <th>FLUX.4</th>
      <th>FLUX.5</th>
      <th>FLUX.6</th>
      <th>FLUX.7</th>
      <th>FLUX.8</th>
      <th>FLUX.9</th>
      <th>FLUX.10</th>
      <th>FLUX.11</th>
      <th>FLUX.12</th>
      <th>FLUX.13</th>
      <th>FLUX.14</th>
      <th>FLUX.15</th>
      <th>FLUX.16</th>
      <th>FLUX.17</th>
      <th>FLUX.18</th>
      <th>FLUX.19</th>
      <th>FLUX.20</th>
      <th>FLUX.21</th>
      <th>FLUX.22</th>
      <th>FLUX.23</th>
      <th>FLUX.24</th>
      <th>FLUX.25</th>
      <th>FLUX.26</th>
      <th>FLUX.27</th>
      <th>FLUX.28</th>
      <th>FLUX.29</th>
      <th>FLUX.30</th>
      <th>FLUX.31</th>
      <th>FLUX.32</th>
      <th>FLUX.33</th>
      <th>FLUX.34</th>
      <th>FLUX.35</th>
      <th>FLUX.36</th>
      <th>FLUX.37</th>
      <th>FLUX.38</th>
      <th>FLUX.39</th>
      <th>FLUX.40</th>
      <th>FLUX.41</th>
      <th>FLUX.42</th>
      <th>FLUX.43</th>
      <th>FLUX.44</th>
      <th>FLUX.45</th>
      <th>FLUX.46</th>
      <th>FLUX.47</th>
      <th>FLUX.48</th>
      <th>...</th>
      <th>FLUX.3149</th>
      <th>FLUX.3150</th>
      <th>FLUX.3151</th>
      <th>FLUX.3152</th>
      <th>FLUX.3153</th>
      <th>FLUX.3154</th>
      <th>FLUX.3155</th>
      <th>FLUX.3156</th>
      <th>FLUX.3157</th>
      <th>FLUX.3158</th>
      <th>FLUX.3159</th>
      <th>FLUX.3160</th>
      <th>FLUX.3161</th>
      <th>FLUX.3162</th>
      <th>FLUX.3163</th>
      <th>FLUX.3164</th>
      <th>FLUX.3165</th>
      <th>FLUX.3166</th>
      <th>FLUX.3167</th>
      <th>FLUX.3168</th>
      <th>FLUX.3169</th>
      <th>FLUX.3170</th>
      <th>FLUX.3171</th>
      <th>FLUX.3172</th>
      <th>FLUX.3173</th>
      <th>FLUX.3174</th>
      <th>FLUX.3175</th>
      <th>FLUX.3176</th>
      <th>FLUX.3177</th>
      <th>FLUX.3178</th>
      <th>FLUX.3179</th>
      <th>FLUX.3180</th>
      <th>FLUX.3181</th>
      <th>FLUX.3182</th>
      <th>FLUX.3183</th>
      <th>FLUX.3184</th>
      <th>FLUX.3185</th>
      <th>FLUX.3186</th>
      <th>FLUX.3187</th>
      <th>FLUX.3188</th>
      <th>FLUX.3189</th>
      <th>FLUX.3190</th>
      <th>FLUX.3191</th>
      <th>FLUX.3192</th>
      <th>FLUX.3193</th>
      <th>FLUX.3194</th>
      <th>FLUX.3195</th>
      <th>FLUX.3196</th>
      <th>FLUX.3197</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2</td>
      <td>119.88</td>
      <td>100.21</td>
      <td>86.46</td>
      <td>48.68</td>
      <td>46.12</td>
      <td>39.39</td>
      <td>18.57</td>
      <td>6.98</td>
      <td>6.63</td>
      <td>-21.97</td>
      <td>-23.17</td>
      <td>-29.26</td>
      <td>-33.99</td>
      <td>-6.25</td>
      <td>-28.12</td>
      <td>-27.24</td>
      <td>-32.28</td>
      <td>-12.29</td>
      <td>-16.57</td>
      <td>-23.86</td>
      <td>-5.69</td>
      <td>9.24</td>
      <td>35.52</td>
      <td>81.20</td>
      <td>116.49</td>
      <td>133.99</td>
      <td>148.97</td>
      <td>174.15</td>
      <td>187.77</td>
      <td>215.30</td>
      <td>246.80</td>
      <td>-56.68</td>
      <td>-56.68</td>
      <td>-56.68</td>
      <td>-52.05</td>
      <td>-31.52</td>
      <td>-31.15</td>
      <td>-48.53</td>
      <td>-38.93</td>
      <td>-26.06</td>
      <td>6.63</td>
      <td>29.13</td>
      <td>64.70</td>
      <td>79.74</td>
      <td>12.21</td>
      <td>12.21</td>
      <td>-19.94</td>
      <td>-28.60</td>
      <td>...</td>
      <td>-11.44</td>
      <td>-21.86</td>
      <td>-16.38</td>
      <td>-7.24</td>
      <td>22.69</td>
      <td>22.69</td>
      <td>7.10</td>
      <td>3.45</td>
      <td>6.49</td>
      <td>-2.55</td>
      <td>12.26</td>
      <td>-7.06</td>
      <td>-23.53</td>
      <td>2.54</td>
      <td>30.21</td>
      <td>38.87</td>
      <td>-22.86</td>
      <td>-22.86</td>
      <td>-4.37</td>
      <td>2.27</td>
      <td>-16.27</td>
      <td>-30.84</td>
      <td>-7.21</td>
      <td>-4.27</td>
      <td>13.60</td>
      <td>15.62</td>
      <td>31.96</td>
      <td>49.89</td>
      <td>86.93</td>
      <td>86.93</td>
      <td>42.99</td>
      <td>48.76</td>
      <td>22.82</td>
      <td>32.79</td>
      <td>30.76</td>
      <td>14.55</td>
      <td>10.92</td>
      <td>22.68</td>
      <td>5.91</td>
      <td>14.52</td>
      <td>19.29</td>
      <td>14.44</td>
      <td>-1.62</td>
      <td>13.33</td>
      <td>45.50</td>
      <td>31.93</td>
      <td>35.78</td>
      <td>269.43</td>
      <td>57.72</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2</td>
      <td>5736.59</td>
      <td>5699.98</td>
      <td>5717.16</td>
      <td>5692.73</td>
      <td>5663.83</td>
      <td>5631.16</td>
      <td>5626.39</td>
      <td>5569.47</td>
      <td>5550.44</td>
      <td>5458.80</td>
      <td>5329.39</td>
      <td>5191.38</td>
      <td>5031.39</td>
      <td>4769.89</td>
      <td>4419.66</td>
      <td>4218.92</td>
      <td>3924.73</td>
      <td>3605.30</td>
      <td>3326.55</td>
      <td>3021.20</td>
      <td>2800.61</td>
      <td>2474.48</td>
      <td>2258.33</td>
      <td>1951.69</td>
      <td>1749.86</td>
      <td>1585.38</td>
      <td>1575.48</td>
      <td>1568.41</td>
      <td>1661.08</td>
      <td>1977.33</td>
      <td>2425.62</td>
      <td>2889.61</td>
      <td>3847.64</td>
      <td>3847.64</td>
      <td>3741.20</td>
      <td>3453.47</td>
      <td>3202.61</td>
      <td>2923.73</td>
      <td>2694.84</td>
      <td>2474.22</td>
      <td>2195.09</td>
      <td>1962.83</td>
      <td>1705.44</td>
      <td>1468.27</td>
      <td>3730.77</td>
      <td>3730.77</td>
      <td>3833.30</td>
      <td>3822.06</td>
      <td>...</td>
      <td>-971.42</td>
      <td>-1327.75</td>
      <td>-1864.69</td>
      <td>-2148.34</td>
      <td>1166.45</td>
      <td>1166.45</td>
      <td>934.66</td>
      <td>574.19</td>
      <td>-216.31</td>
      <td>-3470.75</td>
      <td>-4510.72</td>
      <td>-5013.41</td>
      <td>-3636.05</td>
      <td>-2324.27</td>
      <td>-2688.55</td>
      <td>-2813.66</td>
      <td>-586.22</td>
      <td>-586.22</td>
      <td>-756.80</td>
      <td>-1090.23</td>
      <td>-1388.61</td>
      <td>-1745.36</td>
      <td>-2015.28</td>
      <td>-2359.06</td>
      <td>-2516.66</td>
      <td>-2699.31</td>
      <td>-2777.55</td>
      <td>-2732.97</td>
      <td>1167.39</td>
      <td>1167.39</td>
      <td>1368.89</td>
      <td>1434.80</td>
      <td>1360.75</td>
      <td>1148.44</td>
      <td>1117.67</td>
      <td>714.86</td>
      <td>419.02</td>
      <td>57.06</td>
      <td>-175.66</td>
      <td>-581.91</td>
      <td>-984.09</td>
      <td>-1230.89</td>
      <td>-1600.45</td>
      <td>-1824.53</td>
      <td>-2061.17</td>
      <td>-2265.98</td>
      <td>-2366.19</td>
      <td>-2294.86</td>
      <td>-2034.72</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>844.48</td>
      <td>817.49</td>
      <td>770.07</td>
      <td>675.01</td>
      <td>605.52</td>
      <td>499.45</td>
      <td>440.77</td>
      <td>362.95</td>
      <td>207.27</td>
      <td>150.46</td>
      <td>85.49</td>
      <td>-20.12</td>
      <td>-35.88</td>
      <td>-65.59</td>
      <td>-15.12</td>
      <td>16.60</td>
      <td>-25.70</td>
      <td>61.88</td>
      <td>53.18</td>
      <td>64.32</td>
      <td>72.38</td>
      <td>100.35</td>
      <td>67.26</td>
      <td>14.71</td>
      <td>-16.41</td>
      <td>-147.46</td>
      <td>-231.27</td>
      <td>-320.29</td>
      <td>-407.82</td>
      <td>-450.48</td>
      <td>-146.99</td>
      <td>-146.99</td>
      <td>-146.99</td>
      <td>-146.99</td>
      <td>-166.30</td>
      <td>-139.90</td>
      <td>-96.41</td>
      <td>-23.49</td>
      <td>13.59</td>
      <td>67.59</td>
      <td>32.09</td>
      <td>76.65</td>
      <td>58.30</td>
      <td>5.41</td>
      <td>61.66</td>
      <td>61.66</td>
      <td>126.79</td>
      <td>20.80</td>
      <td>...</td>
      <td>-28.46</td>
      <td>-38.15</td>
      <td>-61.43</td>
      <td>-127.18</td>
      <td>-12.15</td>
      <td>-12.15</td>
      <td>-80.84</td>
      <td>-112.96</td>
      <td>-129.34</td>
      <td>-35.24</td>
      <td>-70.13</td>
      <td>-35.30</td>
      <td>-56.48</td>
      <td>-74.60</td>
      <td>-115.18</td>
      <td>-8.91</td>
      <td>-37.59</td>
      <td>-37.59</td>
      <td>-37.43</td>
      <td>-104.23</td>
      <td>-101.45</td>
      <td>-107.35</td>
      <td>-109.82</td>
      <td>-126.27</td>
      <td>-170.32</td>
      <td>-117.85</td>
      <td>-32.30</td>
      <td>-70.18</td>
      <td>314.29</td>
      <td>314.29</td>
      <td>314.29</td>
      <td>149.71</td>
      <td>54.60</td>
      <td>12.60</td>
      <td>-133.68</td>
      <td>-78.16</td>
      <td>-52.30</td>
      <td>-8.55</td>
      <td>-19.73</td>
      <td>17.82</td>
      <td>-51.66</td>
      <td>-48.29</td>
      <td>-59.99</td>
      <td>-82.10</td>
      <td>-174.54</td>
      <td>-95.23</td>
      <td>-162.68</td>
      <td>-36.79</td>
      <td>30.63</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2</td>
      <td>-826.00</td>
      <td>-827.31</td>
      <td>-846.12</td>
      <td>-836.03</td>
      <td>-745.50</td>
      <td>-784.69</td>
      <td>-791.22</td>
      <td>-746.50</td>
      <td>-709.53</td>
      <td>-679.56</td>
      <td>-706.03</td>
      <td>-720.56</td>
      <td>-631.12</td>
      <td>-659.16</td>
      <td>-672.03</td>
      <td>-665.06</td>
      <td>-667.94</td>
      <td>-660.84</td>
      <td>-672.75</td>
      <td>-644.91</td>
      <td>-680.53</td>
      <td>-620.50</td>
      <td>-570.34</td>
      <td>-530.00</td>
      <td>-537.88</td>
      <td>-578.38</td>
      <td>-532.34</td>
      <td>-532.38</td>
      <td>-491.03</td>
      <td>-485.03</td>
      <td>-427.19</td>
      <td>-380.84</td>
      <td>-329.50</td>
      <td>-286.91</td>
      <td>-283.81</td>
      <td>-298.19</td>
      <td>-271.03</td>
      <td>-268.50</td>
      <td>-209.56</td>
      <td>-180.44</td>
      <td>-136.25</td>
      <td>-136.22</td>
      <td>-68.03</td>
      <td>2.88</td>
      <td>-732.94</td>
      <td>-732.94</td>
      <td>-613.06</td>
      <td>-591.62</td>
      <td>...</td>
      <td>-128.00</td>
      <td>-219.88</td>
      <td>-247.56</td>
      <td>-287.50</td>
      <td>-135.41</td>
      <td>-135.41</td>
      <td>40.19</td>
      <td>81.06</td>
      <td>110.88</td>
      <td>16.50</td>
      <td>-1286.59</td>
      <td>-1286.59</td>
      <td>-1286.59</td>
      <td>-1286.59</td>
      <td>-1286.59</td>
      <td>-1286.59</td>
      <td>-1286.59</td>
      <td>-1286.59</td>
      <td>-14.94</td>
      <td>64.09</td>
      <td>8.38</td>
      <td>45.31</td>
      <td>100.72</td>
      <td>91.53</td>
      <td>46.69</td>
      <td>20.34</td>
      <td>30.94</td>
      <td>-36.81</td>
      <td>-33.28</td>
      <td>-69.62</td>
      <td>-208.00</td>
      <td>-280.28</td>
      <td>-340.41</td>
      <td>-337.41</td>
      <td>-268.03</td>
      <td>-245.00</td>
      <td>-230.62</td>
      <td>-129.59</td>
      <td>-35.47</td>
      <td>122.34</td>
      <td>93.03</td>
      <td>93.03</td>
      <td>68.81</td>
      <td>9.81</td>
      <td>20.75</td>
      <td>20.25</td>
      <td>-120.81</td>
      <td>-257.56</td>
      <td>-215.41</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2</td>
      <td>-39.57</td>
      <td>-15.88</td>
      <td>-9.16</td>
      <td>-6.37</td>
      <td>-16.13</td>
      <td>-24.05</td>
      <td>-0.90</td>
      <td>-45.20</td>
      <td>-5.04</td>
      <td>14.62</td>
      <td>-19.52</td>
      <td>-11.43</td>
      <td>-49.80</td>
      <td>25.84</td>
      <td>11.62</td>
      <td>3.18</td>
      <td>-9.59</td>
      <td>14.49</td>
      <td>8.82</td>
      <td>32.32</td>
      <td>-28.90</td>
      <td>-28.90</td>
      <td>-14.09</td>
      <td>-30.87</td>
      <td>-18.99</td>
      <td>-38.60</td>
      <td>-27.79</td>
      <td>9.65</td>
      <td>29.60</td>
      <td>7.88</td>
      <td>42.87</td>
      <td>27.59</td>
      <td>27.05</td>
      <td>20.26</td>
      <td>29.48</td>
      <td>9.71</td>
      <td>22.84</td>
      <td>25.99</td>
      <td>-667.55</td>
      <td>-1336.24</td>
      <td>-1207.88</td>
      <td>-310.07</td>
      <td>6.18</td>
      <td>18.24</td>
      <td>48.23</td>
      <td>7.60</td>
      <td>34.93</td>
      <td>20.13</td>
      <td>...</td>
      <td>-28.68</td>
      <td>62.41</td>
      <td>93.07</td>
      <td>-217.29</td>
      <td>-217.29</td>
      <td>-217.29</td>
      <td>-217.29</td>
      <td>-203.96</td>
      <td>-171.62</td>
      <td>-122.12</td>
      <td>-32.01</td>
      <td>-47.15</td>
      <td>-56.45</td>
      <td>-41.71</td>
      <td>-34.13</td>
      <td>-43.12</td>
      <td>-53.63</td>
      <td>-53.63</td>
      <td>-53.63</td>
      <td>-24.29</td>
      <td>22.29</td>
      <td>25.18</td>
      <td>1.84</td>
      <td>-22.29</td>
      <td>-26.43</td>
      <td>-12.12</td>
      <td>-33.05</td>
      <td>-21.66</td>
      <td>-228.32</td>
      <td>-228.32</td>
      <td>-228.32</td>
      <td>-187.35</td>
      <td>-166.23</td>
      <td>-115.54</td>
      <td>-50.18</td>
      <td>-37.96</td>
      <td>-22.37</td>
      <td>-4.74</td>
      <td>-35.82</td>
      <td>-37.87</td>
      <td>-61.85</td>
      <td>-27.15</td>
      <td>-21.18</td>
      <td>-33.76</td>
      <td>-85.34</td>
      <td>-81.46</td>
      <td>-61.98</td>
      <td>-69.34</td>
      <td>-17.84</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 3198 columns</p>
</div>


Our target column `LABEL` assigns each star with a 1 or a 2 to designate whether or not there is a confirmed exoplanet that was found in the star's orbit. This is precisely what we are trying to classify in our model below.

Notice there are a total of only 42 stars that are labeled "2", ie confirmed exoplanet orbiting this star. 
There are 37 exoplanet host stars in the training set, and only 5 in the test set. Such highly imbalanced classes will be something we need to deal with carefully in our model.


```python
# Check the value counts 
display(train['LABEL'].value_counts(),test['LABEL'].value_counts())
```


    1    5050
    2      37
    Name: LABEL, dtype: int64



    1    565
    2      5
    Name: LABEL, dtype: int64


Much of the heavy lifting has already been applied to this data set, at least as far as there not being any missing data...


```python
# check for nulls
print('Train Nulls:',train.isna().sum().value_counts())
print('Test Nulls:',test.isna().sum().value_counts())
```

    Train Nulls: 0    3198
    dtype: int64
    Test Nulls: 0    3198
    dtype: int64


# Explore

## Planet Host vs Non-Host Stars

Since we are setting out to classify stars as being either a planet-host or non-host, it would be useful to compare the data visually and see if we can pick up on any significant differences in the flux values. The simplest way to do this is plot the signals of each type as a scatter plot and a line plot.

### Threshold Crossing Event (TCE)
TCE is determined by a significant dip in the flux values, the assumption being something crossed in front of the star blocking its light and this could be an orbiting planet! 


```python
# grab first row of observations to create pandas series 
# first row is label = 2 which is a confirmed exoplanet host star
star_signal_alpha = train.iloc[0, :]
# last row is label = 1 which shows no sign of a TCE (threshold crossing event)
star_signal_beta = train.iloc[-1, :]

display(star_signal_alpha.head(),star_signal_beta.head())
```


    LABEL      2.00
    FLUX.1    93.85
    FLUX.2    83.81
    FLUX.3    20.10
    FLUX.4   -26.98
    Name: 0, dtype: float64



    LABEL       1.00
    FLUX.1    323.28
    FLUX.2    306.36
    FLUX.3    293.16
    FLUX.4    287.67
    Name: 5086, dtype: float64



```python
def star_signals(signal, label_col=None, classes=None, 
                 class_names=None, figsize=(15,5), y_units=None, x_units=None):
    """
    Plots a scatter plot and line plot of time series signal values.  
    
    **ARGS
    signal: pandas series or numpy array
    label_col: name of the label column if using labeled pandas series
        -use default None for numpy array or unlabeled series.
        -this is simply for customizing plot Title to include classification    
    classes: (optional- req labeled data) tuple if binary, array if multiclass
    class_names: tuple or array of strings denoting what the classes mean
    figsize: size of the figures (default = (15,5))
    ******
    
    Ex1: Labeled timeseries passing 1st row of pandas dataframe
    > first create the signal:
    star_signal_alpha = train.iloc[0, :]
    > then plot:
    star_signals(star_signal_alpha, label_col='LABEL',classes=[1,2], 
                 class_names=['No Planet', 'Planet']), figsize=(15,5))
    
    
    Ex2: numpy array without any labels
    > first create the signal:
    
    >then plot:
    star_signals(signal, figsize=(15,5))
    """
    
    # pass None to label_col if unlabeled data, creates generic title
    if label_col is None:
        label = None
        title_scatter = "Scatterplot of Star Flux Signals"
        title_line = "Line Plot of Star Flux Signals"
        color='black'
        
    # store target column as variable 
    elif label_col is not None:
        label = signal[label_col]
        # for labeled timeseries
        if label == 1:
            cls = classes[0]
            cn = class_names[0]
            color='red'

        elif label == 2:
            cls = classes[1]
            cn = class_names[1] 
            color='blue'
    #create appropriate title acc to class_names    
        title_scatter = f"Scatterplot for Star Flux Signal: {cn}"
        title_line = f"Line Plot for Star Flux Signal: {cn}"
    
    # Set x and y axis labels according to units
    # if the units are unknown, we will default to "Flux"
    if y_units == None:
        y_units = 'Flux'
    else:
        y_units = y_units
    # it is assumed this is a timeseries, default to "time"   
    if x_units == None:
        x_units = 'Time'
    else:
        x_units = x_units
    
    # Scatter Plot 
    
    plt.figure(figsize=figsize)
    plt.scatter(pd.Series([i for i in range(1, len(signal))]), 
                signal[1:], marker=4, color=color, alpha=0.7)
    plt.ylabel(y_units)
    plt.xlabel(x_units)
    plt.title(title_scatter)
    plt.show();

    # Line Plot
    plt.figure(figsize=figsize)
    plt.plot(pd.Series([i for i in range(1, len(signal))]), 
             signal[1:], color=color, alpha=0.7)
    plt.ylabel(y_units)
    plt.xlabel(x_units)
    plt.title(title_line)
    plt.show();
```

# A Word on Units..

After doing a little research (mostly by reading the K2 Handbook and visiting the MAST website where NASA houses all of its space telescope data) we learn that the flux values for campaign 3 that are in the Kaggle dataset have been put through a de-noising process. Prior to this particular de-noising process, the flux values would be called `SAP Flux` however in this case we are dealing with `PDC_SAP Flux`. At the moment the units may not seem to matter much, since we assume they are consist across all observations. However, as with anything relating to physics, and science for that matter, the units MATTER. All that to say, for now we are at least going to label the axes accurately so that later down the line if we want to compare this dataset to another from the archive, we will know the units! :)


```python
# plot scatterplots and line plots for both signals

star_signals(signal=star_signal_alpha, label_col='LABEL', classes=[1,2], 
                 class_names=['No Planet', 'Planet'], figsize=(13,5), 
             y_units='PDC_SAP Flux', x_units='Time')

star_signals(signal=star_signal_beta, label_col='LABEL', classes=[1,2], 
                 class_names=['No Planet', 'Planet'], figsize=(13,5),
             y_units='PDC_SAP Flux', x_units='Time')
```


![png](output_25_0.png)



![png](output_25_1.png)



![png](output_25_2.png)



![png](output_25_3.png)


# Model


```python
# import additional libraries from sklearn
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
from scipy.ndimage.filters import uniform_filter1d
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
```


```python
# !pip install keras
# !pip install tensorflow
```


```python
# import additional libraries for keras
import keras
from keras.utils.np_utils import to_categorical

# from keras.preprocessing.text import Tokenizer
from keras import models, layers, optimizers


from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, \
BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam
```

## Train-Test Split


```python
# Using Numpy instead of Pandas to create the 1-dimensional arrays
def numpy_train_test_split(data_folder, train_set, test_set):
    """
    create target classes for training and test data using numpy
    """
    import numpy as np
    
    train = np.loadtxt(data_folder+train_set, skiprows=1, delimiter=',')
    x_train = train[:, 1:]
    y_train = train[:, 0, np.newaxis] - 1.
    
    test = np.loadtxt(data_folder+test_set, skiprows=1, delimiter=',')
    x_test = test[:, 1:]
    y_test = test[:, 0, np.newaxis] - 1.
    
    train,test
    
    return x_train, y_train, x_test, y_test
```


```python
x_train, y_train, x_test, y_test = numpy_train_test_split(data_folder='data/', 
                                                          train_set='exoTrain.csv', 
                                                          test_set='exoTest.csv')
```

## Scaling

Scale each observation to zero mean and unit variance.


```python
def zero_scaler(x_train, x_test):
    """
    Scales each observation of an array to zero mean and unit variance.
    Takes array for train and test data separately.
    """
    import numpy as np
        
    x_train = ((x_train - np.mean(x_train, axis=1).reshape(-1,1)) / 
           np.std(x_train, axis=1).reshape(-1,1))
    
    x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / 
              np.std(x_test, axis=1).reshape(-1,1))
 
    return x_train, x_test
```


```python
x_train,x_test = zero_scaler(x_train, x_test)
```

## De-noising

In order to reduce the amount of high frequency noise that is likely to have an adverse effect on the neural network's learning outcomes, we can try passing a uniform 1-D filter on our scaled train and test data then stack the arrays along the second axis. There are other ways of accomplishing the de-noising (not to mention the scaling and also normalization), but for now we'll take this approach in order to complete the process of building our initial baseline model.


```python
def time_filter(x_train, x_test, step_size=None, axis=2):
    """
    Adds an input corresponding to the running average over a set number
    of time steps. This helps the neural network to ignore high frequency 
    noise by passing in a uniform 1-D filter and stacking the arrays. 
    
    **ARGS
    step_size: integer, # timesteps for 1D filter. defaults to 200
    axis: which axis to stack the arrays
    """
    import numpy as np
    from scipy.ndimage.filters import uniform_filter1d
    
    if step_size is None:
        step_size=200
    
    train_filter = uniform_filter1d(x_train, axis=1, size=step_size)
    test_filter = uniform_filter1d(x_test, axis=1, size=step_size)
    
    x_train = np.stack([x_train, train_filter], axis=2)
    x_test = np.stack([x_test, test_filter], axis=2)
#     x_train = np.stack([x_train, uniform_filter1d(x_train, axis=1, 
#                                                  size=time_steps)], axis=2)
#     x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, 
#                                                size=time_steps)], axis=2)
    
    return x_train, x_test
```


```python
x_train, x_test = time_filter(x_train, x_test, step_size=200, axis=2)
```


```python
import matplotlib
matplotlib.__version__
```




    '3.1.1'




```python

```


```python
print(x_train.shape)
print(y_train.shape)
```

    (5087, 3197, 2)
    (5087, 1)



```python
print(x_test.shape)
print(y_test.shape)
```

    (570, 3197, 2)
    (570, 1)


## Build Model

### **Tactical Decisions**

Since I'm building the baseline model from scratch, a few considerations need to be made. While we can run a gridsearch (or randomizedsearchCV) to get the parameters for us, we still need to decide what type of model would be most ideal for this dataset, knowing what we know so far based on the work done so far. From there, we can go with best practices, assess the initial outcomes, and tune the hyperparameters with each iteration. 

**CNN**
The baseline will consist of a one-dimensional convolutional neural network (CNN). This is ideal for working with this particular dataset in which we will pass one row of the timeseries flux values as an array. This is very similar to how we would process image data (and that's strategically useful if we want to develop the model in the future to handle Full-Frame Images from Tess, for instance, or spectographs of the flux frequences, for instance. 

**1-Layer at a time**
We'll be using the Keras API which makes it easy to add in the layers one at a time. Each 1D convolutional layer corresponds to a local filter, and then a pooling layer reduces the data length by approximately a factor 4. At the end, there are two dense layers. Again, this is similar to the approach taken for a typical image classifier. 

**Activation Function**
The RELU activation function is closest to how real neurons actually work and often produces the best results compared to the other options, so we'll at least start with this for the baseline.

**Batch Normalization**
Finally, the batch normalization layers are what help to speed up convergence. 


```python
def keras_1D(model=Sequential(), kernel_size=11, 
                           activation='relu', 
                           input_shape=x_train.shape[1:], strides=4):
    """
    Linear neural network model using the Keras API
    """
    model = model
    #layer1: the first layer will receive an input shape
    model.add(Conv1D(filters=8, kernel_size=kernel_size, 
                     activation=activation, input_shape=input_shape))
    model.add(MaxPool1D(strides=strides))
    model.add(BatchNormalization())
    #layer2
    model.add(Conv1D(filters=16, kernel_size=kernel_size, 
                     activation=activation))
    model.add(MaxPool1D(strides=strides))
    model.add(BatchNormalization())
    #layer3
    model.add(Conv1D(filters=32, kernel_size=kernel_size, 
                     activation=activation))
    model.add(MaxPool1D(strides=strides))
    model.add(BatchNormalization())
    #layer4
    model.add(Conv1D(filters=64, kernel_size=kernel_size, 
                     activation=activation))
    model.add(MaxPool1D(strides=strides))
    model.add(Flatten())
    
    #dropout layer1 with automatic shape inference
    model.add(Dropout(0.5))
    model.add(Dense(64, activation=activation))
    #dropout layer2
    model.add(Dropout(0.25))
    model.add(Dense(64, activation=activation))
    # sigmoid layer
    model.add(Dense(1, activation='sigmoid'))
    
    return model
```


```python

# set our build function to use the baseline model we built initially
keras_train = keras_1D(model=Sequential(), kernel_size=11, activation='relu', 
                           input_shape=x_train.shape[1:], strides=4)
```

## Batch Generator

To correct for the extremely unbalanced dataset, we'll ensure that the network sees 50% of the positive sample over each batch. We will also apply augmentation by rotating each of the samples randomly each time, thus generating new data. This is similar to image classification when we rotate or shift the samples each time.


```python
def batch_maker(x_train, y_train, batch_size=32):
    """
    Gives equal number of positive and negative samples rotating randomly
    
    generator: A generator or an instance of `keras.utils.Sequence`
        
    The output of the generator must be either
    - a tuple `(inputs, targets)`
    - a tuple `(inputs, targets, sample_weights)`.

    This tuple (a single output of the generator) makes a single
    batch. Therefore, all arrays in this tuple must have the same
    length (equal to the size of this batch). Different batches may have 
    different sizes. 

    For example, the last batch of the epoch
    is commonly smaller than the others, if the size of the dataset
    is not divisible by the batch size.
    The generator is expected to loop over its data
    indefinitely. An epoch finishes when `steps_per_epoch`
    batches have been seen by the model.
    
    """
    import numpy
    import random

    half_batch = batch_size // 2
    
    # Returns a new array of given shape and type, without initializing entries.
    # x_train.shape = (5087, 3197, 2)
    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')
    
    #y_train.shape = (5087, 1)
    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')
    
    pos_idx = np.where(y_train[:,0] == 1.)[0]
    neg_idx = np.where(y_train[:,0] == 0.)[0]

    # rotating each of the samples randomly
    while True:
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)
    
        x_batch[:half_batch] = x_train[pos_idx[:half_batch]]
        x_batch[half_batch:] = x_train[neg_idx[half_batch:batch_size]]
        y_batch[:half_batch] = y_train[pos_idx[:half_batch]]
        y_batch[half_batch:] = y_train[neg_idx[half_batch:batch_size]]
    
        for i in range(batch_size):
            sz = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], sz, axis = 0)
     
        yield x_batch, y_batch
```

## Train Model


```python
def scikit_keras(build_fn=None, compiler=None, params=None, batch_size=32):
    """
    Builds, compiles and fits a keras model
    Takes in dictionaries of parameters for both compiler and
    fit_generator.
    
    *ARGS
    build_fn: build function for creating model, can also pass in a model
    compiler : dict of paramaters for model.compile()
    params : dict of parameters for model.fit_generator
    note: batch
    
    
    """
    # set default parameters if not made explicit
    
    # BUILD vars
    if build_fn:
        model=build_fn
    else:
        model = keras_1D(model=Sequential(), kernel_size=11, activation='relu', 
                           input_shape=x_train.shape[1:], strides=4)

    # COMPILE vars
    if compiler:   
        optimizer=compiler['optimizer']
        learning_rate=compiler['learning_rate'] 
        loss=compiler['loss']
        metrics=compiler['metrics']
     
    else:
        optimizer=Adam
        learning_rate=1e-5
        loss='binary_crossentropy'
        metrics=['accuracy']
        
        
    ##### COMPILE AND FIT #####
    model.compile(optimizer=optimizer(learning_rate), loss=loss, 
                  metrics=metrics)
    
    # HISTORY vars
#     if generator is None:
#         generator = batch_maker(x_train, y_train, batch_size)
    
    if params:
        validation_data = params['validation_data']
        verbose = params['verbose']
        epochs = params['epochs']
        steps_per_epoch = params['steps_per_epoch']
    else:
        validation_data = (x_test, y_test)
        verbose=0
        epochs=5
        steps_per_epoch=x_train.shape[1]//32
    
    history = model.fit_generator(batch_maker(x_train, y_train, batch_size), 
                                  validation_data=validation_data, 
                                  verbose=verbose, epochs=epochs, 
                                  steps_per_epoch=steps_per_epoch)
    
    return model, history
```

# `Model 1`

We'll begin creating a baseline model with a lower than usual learning rate and then speed things up and fine-tune parameters for optimization in the next iterations. (The lower learning rate will help to ensure convergence.) 

We'll increase the learning rate in Model2 iteration and also tune any other parameters as necessary. The first iteration uses the Adam optimizer, however, SGD is also a good option we could try here.


```python
# create params dict for compiling model
compiler = dict(optimizer=Adam,
                learning_rate=1e-5,
                loss='binary_crossentropy',
                metrics=['accuracy'])

# create dict for fit_generator parameters
params = dict(validation_data = (x_test, y_test), 
              verbose=0, 
              epochs=5, 
              steps_per_epoch=(x_train.shape[1]//32))
```


```python
# MODEL 1
# using the baseline model as our build model function
m1, h1 = scikit_keras(build_fn=keras_train, compiler=compiler, params=params)
```


```python
model = m1
```

## Summary (M1)


```python
m1.summary()
```

    Model: "sequential_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv1d_9 (Conv1D)            (None, 3187, 8)           184       
    _________________________________________________________________
    max_pooling1d_9 (MaxPooling1 (None, 797, 8)            0         
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 797, 8)            32        
    _________________________________________________________________
    conv1d_10 (Conv1D)           (None, 787, 16)           1424      
    _________________________________________________________________
    max_pooling1d_10 (MaxPooling (None, 197, 16)           0         
    _________________________________________________________________
    batch_normalization_8 (Batch (None, 197, 16)           64        
    _________________________________________________________________
    conv1d_11 (Conv1D)           (None, 187, 32)           5664      
    _________________________________________________________________
    max_pooling1d_11 (MaxPooling (None, 47, 32)            0         
    _________________________________________________________________
    batch_normalization_9 (Batch (None, 47, 32)            128       
    _________________________________________________________________
    conv1d_12 (Conv1D)           (None, 37, 64)            22592     
    _________________________________________________________________
    max_pooling1d_12 (MaxPooling (None, 9, 64)             0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 576)               0         
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 576)               0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 64)                36928     
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_9 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 71,241
    Trainable params: 71,129
    Non-trainable params: 112
    _________________________________________________________________


## Class Predictions

### Probability Values


```python
# the probability values of the predictions
# these need to be converted into binary values to be understood as far 
# what the class predictions are
y_hat = m1.predict(x_test)[:,0] 
y_hat
```




    array([0.530919  , 0.51738626, 0.524384  , 0.41321602, 0.5511563 ,
           0.32795173, 0.5106873 , 0.5612117 , 0.43855792, 0.4804491 ,
           0.39555854, 0.50140166, 0.4499296 , 0.7180451 , 0.54435474,
           0.47237363, 0.6119266 , 0.5292753 , 0.290318  , 0.38759863,
           0.3710879 , 0.62837756, 0.56779474, 0.58072776, 0.44248626,
           0.50658584, 0.53946865, 0.5768624 , 0.63863194, 0.3791237 ,
           0.39333433, 0.46123534, 0.38098073, 0.47143   , 0.40027162,
           0.4813469 , 0.55804807, 0.46496287, 0.24094132, 0.507898  ,
           0.42323443, 0.22893941, 0.5532119 , 0.2630713 , 0.40238202,
           0.42992753, 0.44642058, 0.4357788 , 0.36307332, 0.14127928,
           0.53372216, 0.5907407 , 0.5476103 , 0.59997076, 0.5050995 ,
           0.32312074, 0.4800474 , 0.33377615, 0.4565567 , 0.57898235,
           0.5781783 , 0.5695022 , 0.29058912, 0.5217019 , 0.34660065,
           0.46119708, 0.32472047, 0.3505417 , 0.4313317 , 0.5275557 ,
           0.5705383 , 0.39403045, 0.5253105 , 0.37416577, 0.30179363,
           0.562756  , 0.49676982, 0.45445392, 0.567758  , 0.5680137 ,
           0.8419337 , 0.5707284 , 0.54805523, 0.43754303, 0.5936526 ,
           0.47762176, 0.39361605, 0.546281  , 0.39019343, 0.6279365 ,
           0.4623438 , 0.41777033, 0.47337162, 0.47508505, 0.29920155,
           0.41973737, 0.53231364, 0.5145746 , 0.5149667 , 0.5648742 ,
           0.6222489 , 0.5817654 , 0.40659297, 0.6101701 , 0.48071346,
           0.39333174, 0.49112976, 0.5407175 , 0.41483283, 0.46194452,
           0.4454074 , 0.55084115, 0.5792704 , 0.563247  , 0.6078335 ,
           0.27323055, 0.54219943, 0.44106632, 0.6646909 , 0.31875217,
           0.47332633, 0.48067713, 0.40319848, 0.3634749 , 0.43528193,
           0.4219565 , 0.38527292, 0.6297061 , 0.5130146 , 0.3635804 ,
           0.4736248 , 0.44121456, 0.47227383, 0.53858054, 0.347077  ,
           0.44604397, 0.4702692 , 0.4724062 , 0.6089753 , 0.49763557,
           0.34900844, 0.46844128, 0.4130219 , 0.28097805, 0.47137994,
           0.5914181 , 0.55949146, 0.557135  , 0.48677886, 0.31485868,
           0.4186572 , 0.41888222, 0.49304315, 0.44315627, 0.4234315 ,
           0.41015708, 0.5902552 , 0.773598  , 0.53177255, 0.45687202,
           0.4048917 , 0.5942385 , 0.5527816 , 0.5877691 , 0.40582684,
           0.35814202, 0.68383604, 0.51380956, 0.48576868, 0.34394133,
           0.3990341 , 0.60928905, 0.67196566, 0.6013788 , 0.637428  ,
           0.45066082, 0.36395174, 0.4398475 , 0.46660012, 0.45963523,
           0.57647014, 0.40196872, 0.28896806, 0.5634549 , 0.48839867,
           0.48565426, 0.37893373, 0.39569312, 0.44585478, 0.4500008 ,
           0.5053754 , 0.39722264, 0.5305519 , 0.4981261 , 0.5014659 ,
           0.33820134, 0.5528475 , 0.45158616, 0.4454329 , 0.4270645 ,
           0.47141126, 0.46171156, 0.5276759 , 0.54669404, 0.4343969 ,
           0.48704132, 0.38517582, 0.46875873, 0.5267358 , 0.55326134,
           0.4999097 , 0.4719466 , 0.42388704, 0.41146344, 0.4318274 ,
           0.19168049, 0.54047304, 0.4684818 , 0.551927  , 0.51300234,
           0.5242145 , 0.5055268 , 0.49299383, 0.25352335, 0.50457096,
           0.49896523, 0.52514225, 0.40509647, 0.28990847, 0.58784556,
           0.54098445, 0.45309788, 0.61368227, 0.4265983 , 0.49333212,
           0.41180137, 0.21455559, 0.62978303, 0.522638  , 0.47029743,
           0.4008945 , 0.39700788, 0.36331326, 0.42667195, 0.38887358,
           0.4849153 , 0.5410577 , 0.50649714, 0.3843378 , 0.4744056 ,
           0.43912035, 0.36714625, 0.40061313, 0.42582917, 0.46585914,
           0.51432085, 0.34872133, 0.4556672 , 0.62962604, 0.40313035,
           0.40402758, 0.4185383 , 0.52919006, 0.5015177 , 0.41968685,
           0.51341754, 0.3457669 , 0.4715502 , 0.291744  , 0.49079797,
           0.4039695 , 0.392378  , 0.46342656, 0.53437835, 0.41920382,
           0.34527972, 0.54282105, 0.6249156 , 0.6530406 , 0.5314289 ,
           0.4301639 , 0.4043473 , 0.38781428, 0.6141419 , 0.5135738 ,
           0.38781422, 0.5260777 , 0.31878656, 0.6026384 , 0.48507544,
           0.41487476, 0.58390164, 0.5696454 , 0.4995578 , 0.5289438 ,
           0.3334803 , 0.48604858, 0.6245774 , 0.4393736 , 0.47240007,
           0.33763844, 0.57947993, 0.20368522, 0.47009772, 0.51766497,
           0.7176566 , 0.5058288 , 0.4242788 , 0.45705998, 0.3782838 ,
           0.46263254, 0.4648638 , 0.501644  , 0.5651593 , 0.30928218,
           0.4105355 , 0.52520144, 0.57575244, 0.4338764 , 0.627764  ,
           0.35515934, 0.40400964, 0.4092869 , 0.48446652, 0.58481616,
           0.5730518 , 0.5018347 , 0.49830726, 0.3783433 , 0.52771944,
           0.5173367 , 0.54741985, 0.4927027 , 0.48333055, 0.36770523,
           0.22485289, 0.5034591 , 0.60179484, 0.5664701 , 0.46929273,
           0.46042493, 0.39580923, 0.5142538 , 0.48048455, 0.37450477,
           0.37844464, 0.5590663 , 0.80963326, 0.65188736, 0.37743685,
           0.63885504, 0.3671996 , 0.57814723, 0.5074352 , 0.5429252 ,
           0.45912042, 0.5356116 , 0.52242804, 0.4476528 , 0.31841612,
           0.57650477, 0.57776535, 0.5376856 , 0.65111065, 0.2794482 ,
           0.27829754, 0.48420456, 0.5247077 , 0.6575235 , 0.52440715,
           0.52162945, 0.29724365, 0.49711192, 0.5273184 , 0.34857786,
           0.53168195, 0.29712215, 0.49247113, 0.3827273 , 0.40646034,
           0.27978405, 0.41459036, 0.44628868, 0.49007535, 0.4530134 ,
           0.4578588 , 0.52300966, 0.42761984, 0.4477577 , 0.4393317 ,
           0.19172862, 0.31079173, 0.40140194, 0.25360298, 0.38627046,
           0.4974604 , 0.5132009 , 0.5784873 , 0.3754265 , 0.5045037 ,
           0.38206843, 0.6130146 , 0.576416  , 0.45264593, 0.4842914 ,
           0.5957036 , 0.5093199 , 0.48699483, 0.2719541 , 0.3823824 ,
           0.35342053, 0.47539276, 0.29734993, 0.37744662, 0.56287605,
           0.48175532, 0.5076742 , 0.5596242 , 0.32420704, 0.35198346,
           0.859305  , 0.65556103, 0.6082994 , 0.4845547 , 0.4132297 ,
           0.61526686, 0.38513437, 0.4911582 , 0.6853376 , 0.50103873,
           0.11448273, 0.56489646, 0.44453332, 0.5805448 , 0.67893   ,
           0.47536165, 0.45393884, 0.3726728 , 0.31396806, 0.4863143 ,
           0.38945177, 0.45169634, 0.3758399 , 0.44515118, 0.5386666 ,
           0.55616117, 0.4754178 , 0.6826506 , 0.6353088 , 0.37947115,
           0.49204546, 0.4922037 , 0.2852646 , 0.40533572, 0.5365106 ,
           0.5418878 , 0.48900247, 0.4576335 , 0.36913598, 0.5559964 ,
           0.5930519 , 0.48268044, 0.57649606, 0.47620595, 0.6183643 ,
           0.66568005, 0.46538734, 0.39640918, 0.48149458, 0.24496087,
           0.57814217, 0.5069836 , 0.54655856, 0.42068893, 0.6332818 ,
           0.6030253 , 0.6222219 , 0.4042381 , 0.49438944, 0.39655364,
           0.61865515, 0.52901167, 0.51198137, 0.44461834, 0.33789337,
           0.54713374, 0.34890753, 0.52522224, 0.53926116, 0.5017608 ,
           0.4490298 , 0.46975997, 0.3776201 , 0.4670189 , 0.39710754,
           0.6159359 , 0.42030776, 0.46620226, 0.51977444, 0.519538  ,
           0.37776735, 0.30600837, 0.30245295, 0.3931782 , 0.44292757,
           0.41217992, 0.4059553 , 0.5103006 , 0.50096226, 0.5597341 ,
           0.37515497, 0.59198326, 0.4694767 , 0.49695715, 0.55782956,
           0.63943994, 0.4943366 , 0.4292868 , 0.4079436 , 0.5222584 ,
           0.51381683, 0.38934305, 0.5118788 , 0.491339  , 0.6218153 ,
           0.55332446, 0.54774475, 0.6574875 , 0.40906063, 0.3632249 ,
           0.34226826, 0.52135247, 0.5047478 , 0.40969783, 0.49182132,
           0.6013845 , 0.45502377, 0.82070553, 0.43421173, 0.44975138,
           0.6331635 , 0.28044605, 0.46149674, 0.60161275, 0.5623652 ,
           0.42475536, 0.32717937, 0.4227332 , 0.53145444, 0.6230192 ,
           0.52428454, 0.5709387 , 0.57927686, 0.60576844, 0.3231951 ,
           0.28911692, 0.50168836, 0.3406566 , 0.455632  , 0.51891685,
           0.42809212, 0.45480755, 0.531437  , 0.55758953, 0.36155784,
           0.57777935, 0.6549137 , 0.4717369 , 0.43286335, 0.55457586],
          dtype=float32)



### Target Values


```python
# the test set's true values for our target class:
y_true = (y_test[:, 0] + 0.5).astype("int")
y_true
```




    array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])



### Class Predictions


```python
# Generate class predictions for test set
y_pred = m1.predict_classes(x_test).flatten() # class predictions in binary
y_pred
```




    array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1,
           1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0,
           0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1,
           0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
           0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
           0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
           0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0,
           0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0,
           0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
           1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1,
           0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0,
           1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0,
           0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1,
           1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0,
           1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1,
           0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
           0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0,
           1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0,
           0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
           0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1,
           0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1,
           1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1],
          dtype=int32)




```python
# Build these values into a function for efficiency in next model iterations:

def get_preds(x_test,y_test,model=None,**kwargs):
    #y_true = (y_test[:, 0] + 0.5).astype("int") # flatten and make integer
    #y_hat = model.predict(x_test)[:,0] 
    
    y_true = y_test.flatten()
    y_pred = model.predict_classes(x_test).flatten() # class predictions 
    
    
    yhat_val = pd.Series(y_pred).value_counts(normalize=False)
    yhat_pct = pd.Series(y_pred).value_counts(normalize=True)*100

    print(f"y_hat_vals:\n {yhat_val}")
    print("\n")
    print(f"y_pred:\n {yhat_pct}")
    from sklearn import metrics
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_true, y_pred)
    print('\nAccuracy Score:', acc)

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    print("\nConfusion Matrix")
    display(cm)
```


```python
get_preds(x_test,y_test,model=m1)
```

    y_hat_vals:
     0    335
    1    235
    dtype: int64
    
    
    y_pred:
     0    58.77193
    1    41.22807
    dtype: float64
    
    Accuracy Score: 0.5929824561403508
    
    Confusion Matrix



    array([[334, 231],
           [  1,   4]])


## Evaluate (M1)

Let's assess the model thus far before tuning parameters. We'll create a few helper functions for calculating metrics and analyzing results visually. 

### Scores

### Interpret Scores
Not the most promising results. These scores are abysmal, however we are simply working with a baseline and the numbers should (hopefully) improve with some simply tuning of the hyperparameters, specifically with our learning rate and the number of epochs. 

While 79% is far from optimal, we have to look at some other metrics such as recall and F1 to make a true assessment of the model's accuracy. These other metrics are especially important when working with highly imbalanced classes.

### Classification Report

Sci-kit learn has a nice built-in method for evaluating our model:


```python
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, jaccard_score, f1_score, recall_score

report = metrics.classification_report(y_test,y_pred)
print(report)
```

                  precision    recall  f1-score   support
    
             0.0       1.00      0.59      0.74       565
             1.0       0.02      0.80      0.03         5
    
        accuracy                           0.59       570
       macro avg       0.51      0.70      0.39       570
    weighted avg       0.99      0.59      0.74       570
    


## History Metrics

The baseline model is not meant to give us optimal results - the real test will be in our final model below. First let's take a look at some of the visuals to understand what the scores really mean. This will help us decide how to proceed in tuning the model appropriately.


```python
def plot_keras_history(history,figsize=(10,4),subplot_kws={}):
    if hasattr(history,'history'):
        history=history.history
    figsize=(10,4)
    subplot_kws={}

    acc_keys = list(filter(lambda x: 'acc' in x,history.keys()))
    loss_keys = list(filter(lambda x: 'loss' in x,history.keys()))

    fig,axes=plt.subplots(ncols=2,figsize=figsize,**subplot_kws)
    axes = axes.flatten()

    y_labels= ['Accuracy','Loss']
    for a, metric in enumerate([acc_keys,loss_keys]):
        for i in range(len(metric)):
            ax = pd.Series(history[metric[i]],
                        name=metric[i]).plot(ax=axes[a],label=metric[i])
    [ax.legend() for ax in axes]
    [ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True)) for ax in axes]
    [ax.set(xlabel='Epochs') for ax in axes]
    plt.suptitle('Model Training Results',y=1.01)
    plt.tight_layout()
    plt.show()
```


```python
# plot convergence
plot_keras_history(h1)
```


![png](output_76_0.png)


With only a few epochs, and a small learning rate, it's obvious that our training parameters has room for improvement. This is good - we will definitely need to adjust the learning rate. If that doesn't go far enough in producing desired results, we can also try using a different optimizer such as SGD instead of Adam. For now let's lok at what the predictions actually were in plain terms.

## Confusion Matrix


```python
# generate a confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred, labels=[0,1])
cm
```

As always, it is much easier to interpret these numbers in a plot! Better yet, build a function for the plot for reuse later on:


```python
# PLOT Confusion Matrices

def plot_confusion_matrix(cm, classes=None,
                          normalize=False,
                          title='Confusion matrix',cmap=plt.cm.Blues):
    
    import itertools
    # Check if normalize is set to True
    # If so, normalize the raw confusion matrix before visualizing
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    
    fig, ax = plt.subplots(figsize=(10,10))
    #mask = np.zeros_like(cm, dtype=np.bool)
    #idx = np.triu_indices_from(mask)
    
    #mask[idx] = True

    plt.imshow(cm, cmap=cmap, aspect='equal')
    
    # Add title and axis labels 
    plt.title('Confusion Matrix') 
    plt.ylabel('True label') 
    plt.xlabel('Predicted label')
    
    # Add appropriate axis scales
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    #ax.set_ylim(len(cm), -.5,.5)
    
    # Text formatting
    fmt = '.2f' if normalize else 'd'
    # Add labels to each cell
    thresh = cm.max() / 2.
    # iterate thru matrix and append labels  
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='darkgray' if cm[i, j] > thresh else 'black',
                size=14, weight='bold')
    
    # Add a legend
    plt.colorbar()
    plt.show() 
```


```python
# Plot normalized confusion matrix
conf1a = plot_confusion_matrix(cm, classes=['No Planet', 'Planet'], normalize=True,
                      title='Normalized confusion matrix')
conf1a
```


```python
# Plot NON normalized confusion matrix
conf1b = plot_confusion_matrix(cm, classes=['No Planet', 'Planet'], normalize=False,
                      title='Normalized confusion matrix')
conf1b
```

There it is. Our baseline model missed ALL FIVE planets in the test set! It predicted 111 planets in the training set, when we know there were only 37. This is what 80% accuracy gives us. Note the recall score above was 0 - this (as well as F1 and Jaccard, both of which include recall in their calculations) are critical scores for assessing the model. 

## ROC AUC

Plot the ROC area under the curve


```python
def roc_plots(y_test, y_hat):
    from sklearn import metrics
    from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
    y_true = (y_test[:, 0] + 0.5).astype("int")   
    fpr, tpr, thresholds = roc_curve(y_true, y_hat) 
    fpr, tpr, thresholds = roc_curve(y_true, y_hat)

    # Threshold Cutoff for predictions
    crossover_index = np.min(np.where(1.-fpr <= tpr))
    crossover_cutoff = thresholds[crossover_index]
    crossover_specificity = 1.-fpr[crossover_index]
    #print("Crossover at {0:.2f} with specificity {1:.2f}".format(crossover_cutoff, crossover_specificity))
    
    plt.plot(thresholds, 1.-fpr)
    plt.plot(thresholds, tpr)
    plt.title("Crossover at {0:.2f} with specificity {1:.2f}".format(crossover_cutoff, crossover_specificity))

    plt.show()


    plt.plot(fpr, tpr)
    plt.title("ROC area under curve is {0:.2f}".format(roc_auc_score(y_true, y_hat)))
    plt.show()
    
    score = roc_auc_score(y_true,y_hat)
    print("ROC_AUC SCORE:",score)
    #print("ROC area under curve is {0:.2f}".format(roc_auc_score(y_true, y_hat)))
```


```python
# This 
roc_plots(y_test, y_hat)
```

# `Model 2`

Revising the function for training the model and tuning just two parameters: adjust learning rate to 4e-3, and increase epochs to 40. 

## Tuning Parameters

This time we will create dictionaries for plugging in parameters to the model. We could do a grid search, or more likely, a randomsearch from sklearn to find the optimal parameters, but for now let's finish building the function to take in the parameter dictionaries with an adjusted learning rate.


```python
# set our build function to use the baseline model we built initially
keras_train = keras_1D(model=Sequential(), kernel_size=11, activation='relu', 
                           input_shape=x_train.shape[1:], strides=4)
```


```python
# create params dict for compiling model
# adjust learning rate to 4e-3

compiler = dict(optimizer=Adam,
                learning_rate=4e-3,
                loss='binary_crossentropy',
                metrics=['accuracy'])

# create dict for fit_generator parameters
# increase verbose to 2 and number of epochs to 40

params = dict(validation_data = (x_test, y_test), 
              verbose=2, 
              epochs=40, 
              steps_per_epoch=(x_train.shape[1]//32))
```

## Compile/Fit (M2)


```python
# MODEL 2
# using the baseline model as our build model function

m2, h2 = scikit_keras(build_fn=keras_train, compiler=compiler, params=params)
```

## Summary (M2)


```python
model.summary()
```

## Class Predictions

We then use our trained neural network to classify the test set:


```python
y_pred = model.predict_classes(x_test).flatten()
```


```python
y_pred
```

## Validation

Evaluate our model using the same helper functions as before, this time embedded into one single function to handle all the work.


```python
cm = confusion_matrix(y_test, y_pred, labels=[0,1])
cm
```


```python
def evaluate_model(x_test, y_test, history=None):
    
    # make predictons using test set
    y_true = (y_test[:, 0] + 0.5).astype("int") # flatten and make integer
    y_hat = model.predict(x_test)[:,0] 
    y_pred = model.predict_classes(x_test).flatten() # class predictions 
    
    
    #Plot Model Training Results (PLOT KERAS HISTORY)
    from sklearn import metrics
    if y_true.ndim>1:
        y_true = y_true.argmax(axis=1)
    if y_pred.ndim>1:
        y_pred = y_pred.argmax(axis=1)   
    try:    
        if history is not None:
            plot_keras_history(history)
    except:
        pass
    
    # Print CLASSIFICATION REPORT
    num_dashes=20
    print('\n')
    print('---'*num_dashes)
    print('\tCLASSIFICATION REPORT:')
    print('---'*num_dashes)
#     try:
#         print(metrics.classification_report(y_true,y_pred))
         #fig = plot_confusion_matrix((y_true,y_pred))
#     except Exception as e:
#         print(f"[!] Error during model evaluation:\n\t{e}")

    from sklearn import metrics
    report = metrics.classification_report(y_true,y_pred)
    print(report)
    
    # Adding additional metrics not in sklearn's report   
    from sklearn.metrics import jaccard_score
    jaccard = jaccard_score(y_test, y_hat_test)
    print('Jaccard Similarity Score:',jaccard)
    
    
    # CONFUSION MATRIX
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    # Plot normalized confusion matrix
    fig = plot_confusion_matrix(cm, classes=['No Planet', 'Planet'], 
                                normalize=False,                               
                                title='Normalized confusion matrix')
    plt.show()

    
    # ROC Area Under Curve
    roc_plots(y_test, y_hat_test)
    
```


```python
evaluate_model(x_test, y_test, h2)
```

# `MODEL 3`

Another optimizer we can try is SGD instead of Adam - this may produce better outcomes. We will also adjust the learning rate.


```python
from keras.optimizers import SGD
```


```python
# create params dict for compiling model
# adjust learning rate to 4e-3

compiler = dict(optimizer=SGD,
                learning_rate=4e-2,
                loss='binary_crossentropy',
                metrics=['accuracy'])

# create dict for fit_generator parameters
# increase verbose to 2 and number of epochs to 40

params = dict(validation_data = (x_test, y_test), 
              verbose=2, 
              epochs=40, 
              steps_per_epoch=(x_train.shape[1]//32))
```


```python
# MODEL 3: using the baseline model as our build model function

m3, h3 = scikit_keras(build_fn=keras_train, compiler=compiler, params=params)
```


```python
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
cm
```


```python
evaluate_model(x_test, y_test, h3)
```

# `MODEL 4`


```python
# create params dict for compiling model
# adjust learning rate to 4e-3

compiler = dict(optimizer=SGD,
                learning_rate=4e-2,
                loss='binary_crossentropy',
                metrics=['accuracy'])

# create dict for fit_generator parameters
# increase verbose to 2 and number of epochs to 40

params = dict(validation_data = (x_test, y_test), 
              verbose=2, 
              epochs=40, 
              steps_per_epoch=(x_train.shape[1]/23))

# MODEL 4
# using the baseline model as our build model function

m4, h4 = scikit_keras(build_fn=keras_train, compiler=compiler, params=params, 
                      batch_size=23)
```

# Interpret Results

## Conclusion

Above, we were able to identify with 99% accuracy 3 of 5 stars that have an exoplanet in their orbit. 

# Recommendations

While it is possible to create a fairly accurate model for detecting exoplanets using the raw flux values of an imbalanced data set (imbalanced meaning only a few positive examples in a sea of negatives) - it is clear that important information is misclassified. When it comes to astrophysics, we need to be much more accurate than this, and we need to feel like the model is fully reliable. I cannot conclude that this model is adequately reliable for performing an accurate analysis on this type of data.

My recommendations are the following:

   1. Use datasets from the MAST website (via API) to incorporate other calculations of the star's properties as features to be used for classification algorithms. Furthermore, attempt other types of transformations and normalizations on the data before running the model - for instance, apply a Fourier transform.

   2. Combine data from multiple campaigns and perhaps even multiple telescopes (for instance, matching sky coordinates and time intervals between K2, Kepler, and TESS for a batch of stars that have overlapping observations - this would be critical for finding transit periods that are longer than the campaigns of a single telecope's observation period).

   3. Explore using computer vision on not only the Full Frame images we can collect from telescopes like TESS, but also on spectographs of the flux values themselves. The beauty of machine learning is our ability to rely on the computer to pick up very small nuances in differences that we ourselves cannot see with our own eyes. 
   
   4. Explore using autoencoded machine learning algorithms with Restricted Boltzmann Machines - this type of model has proven to be incredibly effective in the image analysis of handwriting as we've seen applied the MNIST dataset - let's find out if the same is true for images of stars, be they the Full Frame Images or spectographs.

# Future Work

To continue this project, I'll take another approach for detecting exoplanets using computer vision to analyze images of spectographs of this same star flux data set. Please go to the notebook `[starskøpe-2]` to see how I use a Restricted Boltzmann Machines neural network model to classify stars as exoplanet hosts using spectograph images of the flux values to find transiting exoplanets. Following this, I will apply the same algorithm to spectographs of Fourier transformed data, as you will see in `[starskøpe-3]`. 

Additional future work following this project will be to develop my "cyberoptic artificial telescope" as a machine learning driven application that any astrophysicist can use to look at a single or collection of stars and have the model classify them according not only to exoplanet predictions, but also predict what type of star it is, and other key properties that would be of interest for astrophysical science applications.

