
"""
TODO:
* enhance get_response func with more flexibility e.g. type of data json, xml, html etc
>>> Use open_api code (BELOW) to add functionality

>>> add destination e.g. AWS S3 container for Hubble, Google Cloud Storage for other 
large datasets


        \         DON'T PANIC       /
         \                         /
          \                       /
           ]                     [    ,'|
           ]                     [   /  |
           ]___               ___[ ,'   |
           ]  ]\             /[  [ |:   |
           ]  ] \           / [  [ |:   |
           ]  ]  ]         [  [  [ |:   |
           ]  ]  ]__     __[  [  [ |:   |
           ]  ]  ] ]\ _ /[ [  [  [ |:   |
           ]  ]  ] ] (#) [ [  [  [ :===='
           ]  ]  ]_].nHn.[_[  [  [
           ]  ]  ]  HHHHH. [  [  [
           ]  ] /   `HH("N  \ [  [
           ]__]/     HHH  "  \[__[
           ]         NNN         [
           ]         N/"         [
           ]         N H         [
          /          N            \
         /           q,            \
        /                           \
"""
# https://stsci.app.box.com/s/4no7430kswla4gsg8bt2avs72k9agpne

# MAST has notebooks with astroquery/API examples:
# •https://github.com/spacetelescope/notebooks/tree/master/notebooks/MASTAstroqueryDocumentation
# •https://astroquery.readthedocs.io/en/latest/mast/mast.html
# •https://astroquery.readthedocs.io/en/latest/Datasets 
# on the AWS Cloud 
# •https://mast-labs.stsci.io/2018/06/hst-public-data-on-aws

# MAST Slack Channel
# MAST Helpdesk: archive@stsci.edu

def get_response(url = 'https://en.wikipedia.org/wiki/Stock_market',timeout=3,
                 verbose=2):
    """Getting and previewing the website urls response.

    Args: 
        url (str): page to get
        timeout (int):  time to delay request
        verbose (0,1,2): controls info display. 1= header, 2=header+status_code

    Returns:
        response (Response)
    """
    import requests

    response = requests.get(url=url, timeout=timeout)
    if verbose>1:
        print('Status code: ',response.status_code)

        if response.status_code==200:
            print('Connection successfull.\n\n')
        else: 
            print('Error. Check status code table.\n\n')    

    if verbose >0:    
        # Print out the contents of a request's response
        print(f"{'---'*20}\n\tContents of Response.items():\n{'---'*20}")

        for k,v in response.headers.items():
            print(f"{k:{25}}: {v:{40}}") # Note: add :{number} inside of a   

    return response
        
response = get_response(verbose=0)
# help(response)



### MAST API 
## Max request = 500,000 Rows

# Example Request 1 - FITS files

from astroquery.mast import Observations
targets = ['M101', 'M57', 'M55']

for t in targets:
    obsTarget = Observations.query_object(t, radius=0.01)
    want = obsTarget['obs_collection'] == 'HST'
    print('Running Analysis on HST %s data for %s' % (obsTarget[want]['instrument_name'][0], t))

    products = Observations.get_product_list(obsTarget[want])
    manifest = Observations.download_products(products[0:1])
    # Run your analysis of the data here


# Example 2
# import observations
from astroquery.mast import Observations

#query object by searching by name
target = "M57"
obsv = Observations.query_object(target, radius = 0.02)
print("Number of observations: %u" % len(obsv))

## Number of observations: 234

# filter table
want = (obsv['obs_collection']== 'HST')
print(obsv[want]['obs_collection', 'filters', 'dataproduct_type', 'calib_level', 'wavelength_region'])

# useful column names 
obs_collection
filters
dataproduct_type
proposal_id
proposal_pi


# Example 3 - filter by observation properties like start time

from astroquery.mast import Observations
obsTable = Observations.query_criteria(filters=["F350LP", "F814W"], proposal_pi="*Burke*", obs_collection="HST")
print("Num obs: %u" % len(obsTable))

>>> Num obs: 205


## t_min and t_max are given as MJD
want = obsTable['t_min'] < 58094.4 #MJD
print(obsTable[want]['obs_collection', 'filters', 'dataproduct_type','t_min','target_name'])


#DOWNLOAD PRODUCTS
# filter list to those you want to download
want = products['productType'] == 'SCIENCE'
manifest = Observations.download_products(products[want])
# returned manifest give the local path
from astropy.io import fits 
fits.info(manifest[0]['Local Path'])


# Example 5: Observations.Catalogs

catalogData = Catalogs.query_object("M10", radius=0.0, catalog="HSC")
catalogData[0:2]['MatchID','MatchRA','MatchDec']

Catalogs.get_hsc_spectra
Catalogs.download_hsc_spectra


# accessing proprietary data
# To get your token:https://auth.mast.stsci.edu
from astroquery.mast import Observations
Observations.login(token="14151875618756187idsjaha")

>>> INFO: MAST API token accepted, welcome User Name [astroquery.mast.core]

>>> sessionInfo = Observations.session_info()
eppn: username@stsci.edu
ezid: uname


# Example - JSON
 {'service':'Mast.Caom.Cone',
  'params':{'ra':254.28746,
            'dec':-4.09933,
            'radius':0.2},
  'format':'json',
  'pagesize':2000,
  'removenullcolumns':True,
  'timeout':30,
  'removecache':True}

#####



from astroquery.mast import Mast

service = 'Mast.Name.Lookup'
params ={'input':"M8",
         'format':'json'}

response = Mast.mashup_request_async(service,params)
result = response[0].json()
print(result)

{'resolvedCoordinate': [{'cacheDate': 'Apr 12, 2017 9:28:24 PM',
                         'cached': True,
                         'canonicalName': 'MESSIER 008',
                         'decl': -24.38017,
                         'objectType': 'Neb',
                         'ra': 270.92194,
                         'resolver': 'NED',
                         'resolverTime': 113,
                         'searchRadius': -1.0,
                         'searchString': 'm8'}],
 'status': ''}


#### OPEN NOTIFY API EXAMPLE

# import urllib2
# import json
# import requests
# # Make a get request to get the latest position of the international space station from the opennotify api.
# # response = requests.get("http://api.open-notify.org/iss-now.json")

# # Print the status code of the response.
# print(response.status_code)
# print(response.text)
# req = urllib2.Request("http://api.open-notify.org/iss-now.json")
# response = urllib2.urlopen(req)

# obj = json.loads(response.read())

# print obj['timestamp']
# print obj['iss_position']['latitude'], obj['data']['iss_position']['latitude']

# # Example prints:
# #   1364795862
# #   -47.36999493 151.738540034

# # Set up the parameters we want to pass to the API.
# # This is the latitude and longitude of New York City.
# parameters = {"lat": 40.71, "lon": -74}

# # Make a get request with the parameters.
# response = requests.get("http://api.open-notify.org/iss-pass.json", params=parameters)
# print(response.status_code)

# # Print the content of the response (the data the server returned)
# print(dict(response.headers))
# print(response.text)


