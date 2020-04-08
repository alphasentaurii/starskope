
# `STARSKØPE`



**Building a Cyberoptic Neural Network Telescope for Astrophysical Object Classification**

> Flatiron School Capstone Project
* `Ru Keïn`
* `Instructor: James Irving PhD`
* `Data Science Full-Time Program`
* `Blog post URL: http://www.hakkeray.com/datascience/2020/03/22/planetX-hunter-classification-algorithms.html`
* `Non-Technical Presentation: Datascience-CAPSTONE-starskope.pdf`

    Note: this project is divided into 3 notebooks:

    starskøpe I : Keras Neural Network Model (this notebook)
    starskøpe II: Computer Vision/Restricted Boltzmann Machines for Spectographs
    starskøpe III: CV/RBMs for Fourier Transformed Spectographs

![](https://github.com/hakkeray/starskope/blob/master/288_planetbleed1600.jpeg)
*image credit: NASA*

# Mission Brief

## ABSTRACT

**Questions**
Can a transiting exoplanet be detected strictly by analyzing the raw flux values of a given star? What kind of normalization, de-noising, and/or unit conversion is necessary in order for the analysis to be accurate? How much signal-to-noise ratio is too much? That is, if the classes are highly imbalanced, for instance only a few planets can be confirmed out of thousands of stars, does the imbalance make for an unreliable machine learning model?

**MODEL 1**
To answer the above questions, I started the analysis with a labeled timeseries dataset from Kaggle posted by NASA for K2's campaign 3 observations in which only 42 confirmed exoplanets are detected in a set of over 5,000 stars. This data is labeled and therefore primed for machine learning, however it has no units, and while it is a timeseries, the actual timestamps are not included. If we compare this to the type of data available on the MAST website where the original K2 data and other space telescope observations are housed, there is a significant amount of information about the target stars that is missing and typically quite useful in terms of modeling and analysis for the purpose of detecting exoplanets. 

**RESULTS**
Using the Keras API to train a neural network, I was able to identify with 88% accuracy the handful of stars that have an exoplanet orbiting around them. 

**RECOMMENDATIONS**
While it is possible to create a fairly accurate model for detecting exoplanets using the raw flux values of an imbalanced data set (imbalanced meaning only a few positive examples in a sea of negatives) - it is clear that important information is misclassified. When it comes to astrophysics, we need to be much more accurate than this, and we need to feel like the model is fully reliable. I cannot conclude that this model is adequately reliable for performing an accurate analysis on this type of data.

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
     'sparknotes.txt',
     'LICENSE',
     'sklearn-sparknotes.ipynb',
     'starskøpe-2.ipynb',
     'models',
     'todo.md',
     'README.md',
     '.gitignore',
     'starskøpe.ipynb',
     '_config.yml',
     '.ipynb_checkpoints',
     'Dont_Panic',
     '.git',
     'DATA',
     'assets',
     'notebooks',
     'pyFunc']




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


```python
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



```python
# Check the value counts - notice there are a total of only 42 stars that are labeled "2"
# which means there is a confirmed exoplanet. 37 in the training set, 5 in the test set.
display(train['LABEL'].value_counts(),test['LABEL'].value_counts())
```


    1    5050
    2      37
    Name: LABEL, dtype: int64



    1    565
    2      5
    Name: LABEL, dtype: int64


Check for nulls:


```python
train.isna().sum().value_counts()
```




    0    3198
    dtype: int64




```python
test.isna().sum().value_counts()
```




    0    3198
    dtype: int64




```python
## Compare: Exoplanet vs Non-exoplanet Host Stars

# look at the first Star in the dataset (label = 2 means confirmed host of exoplanet)
starA = train.iloc[0, :]
starA.head()
```




    LABEL      2.00
    FLUX.1    93.85
    FLUX.2    83.81
    FLUX.3    20.10
    FLUX.4   -26.98
    Name: 0, dtype: float64




```python
# non-exoplanet host star in last row of index (5086 or -1)
starB = train.iloc[-1, :]
starB.head()
```




    LABEL       1.00
    FLUX.1    323.28
    FLUX.2    306.36
    FLUX.3    293.16
    FLUX.4    287.67
    Name: 5086, dtype: float64



# Explore


```python
# Scatter Plot For First Star
plt.figure(figsize=(15, 5))
plt.scatter(pd.Series([i for i in range(1, len(starA))]), starA[1:])
plt.ylabel('Flux')
plt.xlabel('Time')
plt.title('Flux for StarA - exoplanet TCE')
plt.show()

# Line Plot For First Star
plt.figure(figsize=(15, 5))
plt.plot(pd.Series([i for i in range(1, len(starA))]), starA[1:])
plt.ylabel('Flux')
plt.xlabel('Time')
plt.title('Flux for StarA - exoplanet TCE')
plt.show()


# Scatter Plot For Last Star
plt.figure(figsize=(15, 5))
plt.scatter(pd.Series([i for i in range(1, len(starB))]), starB[1:])
plt.ylabel('Flux')
plt.xlabel('Time')
plt.title('Flux for StarB - no planet')
plt.show()

# Line Plot For last Star
plt.figure(figsize=(15, 5))
plt.plot(pd.Series([i for i in range(1, len(starB))]), starB[1:])
plt.ylabel('Flux')
plt.xlabel('Time')
plt.title('Flux for StarB - no planet')
plt.show()

```


![png](output_20_0.png)



![png](output_20_1.png)



![png](output_20_2.png)



![png](output_20_3.png)


# Model 1 - Keras Neural Network


```python
# import additional libraries from sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
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

from keras.preprocessing.text import Tokenizer
from keras import models, layers, optimizers


from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, \
BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam
```

    Using TensorFlow backend.
    /Users/hakkeray/opt/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /Users/hakkeray/opt/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /Users/hakkeray/opt/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /Users/hakkeray/opt/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /Users/hakkeray/opt/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /Users/hakkeray/opt/anaconda3/lib/python3.7/site-packages/tensorflow/python/framework/dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])


## Train-Test Split


```python
# Using Numpy instead of Pandas to create the 1-dimensional arrays
INPUT_LIB = 'data/'
raw_data = np.loadtxt(INPUT_LIB +'exoTrain.csv', skiprows=1, delimiter=',')
x_train = raw_data[:, 1:]
y_train = raw_data[:, 0, np.newaxis] - 1.
raw_data = np.loadtxt(INPUT_LIB + 'exoTest.csv', skiprows=1, delimiter=',')
x_test = raw_data[:, 1:]
y_test = raw_data[:, 0, np.newaxis] - 1.
del raw_data
```


```python
#Scale each observation to zero mean and unit variance.

x_train = ((x_train - np.mean(x_train, axis=1).reshape(-1,1)) / 
           np.std(x_train, axis=1).reshape(-1,1))
x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / 
          np.std(x_test, axis=1).reshape(-1,1))
```

Add an input corresponding to the running average over 200 time steps. This helps the net ignore high frequency noise and instead look at non-local information.


```python
x_train = np.stack([x_train, uniform_filter1d(x_train, axis=1, size=200)], axis=2)
x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, size=200)], axis=2)
```


```python
x_train.shape
```




    (5087, 3197, 2)




```python
x_test.shape
```




    (570, 3197, 2)



## Train Model

Adding the layers one at a time:

Each 1D convolutional layers corresponds to a local filter, and then a pooling layer reduces the data length by approximately a factor 4. At the end, there are two dense layers, just as we would in a typical image classifier. Batch normalization layers speed up convergence.


```python
model = Sequential()
model.add(Conv1D(filters=8, kernel_size=11, activation='relu', input_shape=x_train.shape[1:]))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=16, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=4))
model.add(BatchNormalization())
model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=4))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

The data here is extremely unbalanced, with only a few positive examples. To correct for this, we need to use the positive examples a lot more often, so that the network sees 50% of each over each batches. 

We also want to generate new examples by rotating them randomly each time. This is called augmentation and is similar to when we rotate/shift examples in image classification.


```python
def batch_generator(x_train, y_train, batch_size=32):
    """
    Gives equal number of positive and negative samples, and rotates them randomly in time
    """
    half_batch = batch_size // 2
    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')
    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')
    
    yes_idx = np.where(y_train[:,0] == 1.)[0]
    non_idx = np.where(y_train[:,0] == 0.)[0]
    
    while True:
        np.random.shuffle(yes_idx)
        np.random.shuffle(non_idx)
    
        x_batch[:half_batch] = x_train[yes_idx[:half_batch]]
        x_batch[half_batch:] = x_train[non_idx[half_batch:batch_size]]
        y_batch[:half_batch] = y_train[yes_idx[:half_batch]]
        y_batch[half_batch:] = y_train[non_idx[half_batch:batch_size]]
    
        for i in range(batch_size):
            sz = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], sz, axis = 0)
     
        yield x_batch, y_batch
```

The hyperparameters here are chosen to finish training within the Kernel, rather than to get optimal results. On a GPU, we could probably use a smaller learning rate, and SGD instead of Adam. 


```python
#Start with a slightly lower learning rate, to ensure convergence
model.compile(optimizer=Adam(1e-5), loss = 'binary_crossentropy', metrics=['accuracy'])
hist = model.fit_generator(batch_generator(x_train, y_train, 32), 
                           validation_data=(x_test, y_test), 
                           verbose=0, epochs=5,
                           steps_per_epoch=x_train.shape[1]//32)
```


```python
#Then speed things up a little
model.compile(optimizer=Adam(4e-5), loss = 'binary_crossentropy', metrics=['accuracy'])
hist = model.fit_generator(batch_generator(x_train, y_train, 32), 
                           validation_data=(x_test, y_test), 
                           verbose=2, epochs=40,
                           steps_per_epoch=x_train.shape[1]//32)
```

    Epoch 1/40
     - 7s - loss: 0.7038 - accuracy: 0.5574 - val_loss: 0.6824 - val_accuracy: 0.5772
    Epoch 2/40
     - 6s - loss: 0.6844 - accuracy: 0.5852 - val_loss: 0.6550 - val_accuracy: 0.6246
    Epoch 3/40
     - 6s - loss: 0.6603 - accuracy: 0.6114 - val_loss: 0.6253 - val_accuracy: 0.6719
    Epoch 4/40
     - 6s - loss: 0.6505 - accuracy: 0.6285 - val_loss: 0.6083 - val_accuracy: 0.6965
    Epoch 5/40
     - 6s - loss: 0.6312 - accuracy: 0.6297 - val_loss: 0.5929 - val_accuracy: 0.7263
    Epoch 6/40
     - 6s - loss: 0.6282 - accuracy: 0.6496 - val_loss: 0.5741 - val_accuracy: 0.7351
    Epoch 7/40
     - 6s - loss: 0.6165 - accuracy: 0.6645 - val_loss: 0.5552 - val_accuracy: 0.7526
    Epoch 8/40
     - 6s - loss: 0.6007 - accuracy: 0.6698 - val_loss: 0.5482 - val_accuracy: 0.7544
    Epoch 9/40
     - 6s - loss: 0.5991 - accuracy: 0.6730 - val_loss: 0.5287 - val_accuracy: 0.7649
    Epoch 10/40
     - 6s - loss: 0.5978 - accuracy: 0.6809 - val_loss: 0.5215 - val_accuracy: 0.7614
    Epoch 11/40
     - 6s - loss: 0.5725 - accuracy: 0.6998 - val_loss: 0.5058 - val_accuracy: 0.7842
    Epoch 12/40
     - 6s - loss: 0.5837 - accuracy: 0.6828 - val_loss: 0.4988 - val_accuracy: 0.7807
    Epoch 13/40
     - 6s - loss: 0.5673 - accuracy: 0.7058 - val_loss: 0.4708 - val_accuracy: 0.7947
    Epoch 14/40
     - 6s - loss: 0.5650 - accuracy: 0.7074 - val_loss: 0.4757 - val_accuracy: 0.7965
    Epoch 15/40
     - 6s - loss: 0.5490 - accuracy: 0.7244 - val_loss: 0.4633 - val_accuracy: 0.7982
    Epoch 16/40
     - 6s - loss: 0.5506 - accuracy: 0.7146 - val_loss: 0.4371 - val_accuracy: 0.8193
    Epoch 17/40
     - 6s - loss: 0.5260 - accuracy: 0.7503 - val_loss: 0.4441 - val_accuracy: 0.8105
    Epoch 18/40
     - 6s - loss: 0.5390 - accuracy: 0.7336 - val_loss: 0.4284 - val_accuracy: 0.8211
    Epoch 19/40
     - 6s - loss: 0.5053 - accuracy: 0.7484 - val_loss: 0.4091 - val_accuracy: 0.8298
    Epoch 20/40
     - 8s - loss: 0.5286 - accuracy: 0.7446 - val_loss: 0.4021 - val_accuracy: 0.8246
    Epoch 21/40
     - 7s - loss: 0.5115 - accuracy: 0.7465 - val_loss: 0.4062 - val_accuracy: 0.8211
    Epoch 22/40
     - 7s - loss: 0.4991 - accuracy: 0.7592 - val_loss: 0.3931 - val_accuracy: 0.8281
    Epoch 23/40
     - 7s - loss: 0.4788 - accuracy: 0.7686 - val_loss: 0.3856 - val_accuracy: 0.8351
    Epoch 24/40
     - 7s - loss: 0.4807 - accuracy: 0.7778 - val_loss: 0.3653 - val_accuracy: 0.8386
    Epoch 25/40
     - 7s - loss: 0.4632 - accuracy: 0.7869 - val_loss: 0.3671 - val_accuracy: 0.8404
    Epoch 26/40
     - 7s - loss: 0.4538 - accuracy: 0.7955 - val_loss: 0.3739 - val_accuracy: 0.8316
    Epoch 27/40
     - 8s - loss: 0.4406 - accuracy: 0.8005 - val_loss: 0.3831 - val_accuracy: 0.8298
    Epoch 28/40
     - 8s - loss: 0.4483 - accuracy: 0.7920 - val_loss: 0.3445 - val_accuracy: 0.8439
    Epoch 29/40
     - 7s - loss: 0.4141 - accuracy: 0.8176 - val_loss: 0.3440 - val_accuracy: 0.8456
    Epoch 30/40
     - 7s - loss: 0.4284 - accuracy: 0.8084 - val_loss: 0.3357 - val_accuracy: 0.8404
    Epoch 31/40
     - 7s - loss: 0.4155 - accuracy: 0.8090 - val_loss: 0.3541 - val_accuracy: 0.8404
    Epoch 32/40
     - 7s - loss: 0.4091 - accuracy: 0.8223 - val_loss: 0.3329 - val_accuracy: 0.8491
    Epoch 33/40
     - 7s - loss: 0.3974 - accuracy: 0.8340 - val_loss: 0.3377 - val_accuracy: 0.8491
    Epoch 34/40
     - 8s - loss: 0.3868 - accuracy: 0.8283 - val_loss: 0.3083 - val_accuracy: 0.8579
    Epoch 35/40
     - 6s - loss: 0.3728 - accuracy: 0.8374 - val_loss: 0.2956 - val_accuracy: 0.8719
    Epoch 36/40
     - 8s - loss: 0.3855 - accuracy: 0.8305 - val_loss: 0.3079 - val_accuracy: 0.8614
    Epoch 37/40
     - 6s - loss: 0.3517 - accuracy: 0.8491 - val_loss: 0.3424 - val_accuracy: 0.8439
    Epoch 38/40
     - 7s - loss: 0.3498 - accuracy: 0.8507 - val_loss: 0.3359 - val_accuracy: 0.8491
    Epoch 39/40
     - 6s - loss: 0.3452 - accuracy: 0.8570 - val_loss: 0.3062 - val_accuracy: 0.8649
    Epoch 40/40
     - 6s - loss: 0.3507 - accuracy: 0.8545 - val_loss: 0.2959 - val_accuracy: 0.8719


# Interpret Results

Look at convergence:


```python
plt.plot(hist.history['loss'], color='blue')
plt.plot(hist.history['val_loss'], color='lime')
plt.show()
plt.plot(hist.history['accuracy'], color='blue')
plt.plot(hist.history['val_accuracy'], color='lime')
plt.show()
```


![png](output_42_0.png)



![png](output_42_1.png)


We then use our trained neural network to classify the test set:


```python
non_idx = np.where(y_test[:,0] == 0.)[0]
yes_idx = np.where(y_test[:,0] == 1.)[0]
y_hat = model.predict(x_test)[:,0]
```


```python
plt.plot([y_hat[i] for i in yes_idx], 'bo')
plt.show()
plt.plot([y_hat[i] for i in non_idx], 'go')
plt.show()
```


![png](output_45_0.png)



![png](output_45_1.png)


Not all five positive examples all get 0.95-1.00 score. More than half of the negative examples get score close to zero, however a decent proportion of the negatives also received scores closer to 1. 

## Validation

With the help of Sci-kit Learn, we can now choose an optimal cutoff score for classification. 


```python
y_true = (y_test[:, 0] + 0.5).astype("int")
fpr, tpr, thresholds = roc_curve(y_true, y_hat)
plt.plot(thresholds, 1.-fpr)
plt.plot(thresholds, tpr)
plt.show()
crossover_index = np.min(np.where(1.-fpr <= tpr))
crossover_cutoff = thresholds[crossover_index]
crossover_specificity = 1.-fpr[crossover_index]
print("Crossover at {0:.2f} with specificity {1:.2f}".format(crossover_cutoff, crossover_specificity))
plt.plot(fpr, tpr)
plt.show()
print("ROC area under curve is {0:.2f}".format(roc_auc_score(y_true, y_hat)))
```


![png](output_49_0.png)


    Crossover at 0.89 with specificity 0.97



![png](output_49_2.png)


    ROC area under curve is 0.99


Let's look at the misclassified data (of which there appears to be quite a few, unfortunately!):


```python
false_positives = np.where(y_hat * (1. - y_test) > 0.5)[0]
for i in non_idx:
    if y_hat[i] > crossover_cutoff:
        print(i)
        plt.plot(x_test[i])
        plt.show()
```

    89



![png](output_51_1.png)


    124



![png](output_51_3.png)


    136



![png](output_51_5.png)


    183



![png](output_51_7.png)


    230



![png](output_51_9.png)


    258



![png](output_51_11.png)


    360



![png](output_51_13.png)


    363



![png](output_51_15.png)


    372



![png](output_51_17.png)


    395



![png](output_51_19.png)


    400



![png](output_51_21.png)


    415



![png](output_51_23.png)


    417



![png](output_51_25.png)


    460



![png](output_51_27.png)


    485



![png](output_51_29.png)


    486



![png](output_51_31.png)


    487



![png](output_51_33.png)


# Conclusion

Above, we were able to identify with 88% accuracy the handful of stars that have an exoplanet orbiting around them. 

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



```python

```
