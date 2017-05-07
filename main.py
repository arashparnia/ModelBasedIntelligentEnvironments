



from __future__ import print_function
import time
import requests
import cv2
import operator
import numpy as np
import cognitive_face as CF

import json
import urllib2
import requests
from requests.auth import HTTPBasicAuth
from webcam import get_image


# user = 'Arash'
# password = '12345'
# data = {
#     'type':'command',
#     'param':'switchlight',
#     'idx':'1',
#     'switchcmd':'Off'
# }
#
# req = urllib2.Request('http://jebo.mynetgear.com:8080')
# req.add_header('Content-Type', 'application/json')
# req.add_header('Authorization','Basic QXJhc2g6MTIzNDU=')
#
#
# response = urllib2.urlopen(req, json.dumps(data))
# print(response)



import base64

# encoded = base64.b64encode(b'Arash:12345')
# print (encoded)


# url = 'http://jebo.mynetgear.com:8080'
# payload1 = {
#     "type":"command",
#     "param":"Light/Switch",
#     "idx":"1",
#     "switchcmd":"Off"
# }

#
# r = requests.post(url, data=json.dumps(payload1), headers=headers)
# print(r.url)
# print(r.raw)
# print (r)
# r = requests.get(url, headers=headers)
# print(r.url)
# print(r.raw)
# print (r)




# headers = {'Authorization': 'Basic QXJhc2g6MTIzNDU='}
# request = urllib2.Request("http://jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=8&switchcmd=Off",
#                           headers=headers)
# request = urllib2.Request("http://jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=8&switchcmd=On",
#                           headers=headers)
# contents = urllib2.urlopen(request).read()



# https://api.telegram.org/bot353028451:AAEoXtjebnEbxWLJORYLGhTyie_co3u0HRo/sendmessage?chat_id=277127047&text=test



# print(urllib.urlopen('http://jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=1&switchcmd=On').read())
# urllib2.urlopen('http://arash:12345@jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=1&switchcmd=On')
# urllib2.urlopen('http://arash:12345@jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=1&switchcmd=Off')
# http://arash:12345@jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=1&switchcmd=On

# Import library to display results
import matplotlib.pyplot as plt

# get_ipython().magic('matplotlib inline')
# Display images within Jupyter


# In[2]:

# Variables

# _url = 'https://westus.api.cognitive.microsoft.com/vision/v1/analyses'
# _key = '10cd386b7ebe4e2dabd9c3ab27f2bddb'  # Here you have to paste your primary key
# _maxNumRetries = 10


KEY = '275f1ebbffeb49389decd1afb68e9157'  # Replace with a valid Subscription Key here.
CF.Key.set(KEY)

# img_url = 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'

# result = CF.face.detect(img_url)
# print(result)

# pathToFileInDisk = r'images/f1.jpeg'
# with open(pathToFileInDisk, 'rb') as f:
#     im1 = f.read()
result_webcam = CF.face.detect(get_image(), landmarks=True, attributes='age,gender')

result_arash = CF.face.detect('images/arash.jpg', landmarks=True, attributes='age,gender')

result1 = CF.face.detect('images/f1.jpeg', landmarks=True, attributes='age,gender')
result2 = CF.face.detect('images/f2.jpg', landmarks=True, attributes='age,gender')
result3 = CF.face.detect('images/f3.jpg', landmarks=True, attributes='age,gender')
result4 = CF.face.detect('images/f4.jpg', landmarks=True, attributes='age,gender')
result5 = CF.face.detect('images/f5.jpg', landmarks=True, attributes='age,gender')

print(result_webcam)
print(result_arash)
print(result2)
print(result3)
print(result4)
print(result5)

face_webcam = result_webcam[0]
face_arash = result_arash[0]
face1 = result1[0]
face2 = result2[0]
face3 = result3[0]
face4 = result4[0]
face5 = result5[0]

faceid_webcam = (face_webcam['faceId'])
faceid_arash = (face_arash['faceId'])
faceid1 = (face1['faceId'])
faceid2 = (face2['faceId'])
faceid3 = (face3['faceId'])
faceid4 = (face4['faceId'])
faceid5 = (face5['faceId'])

# print(faceid1)
# print(faceid2)

# CF.face_list.create('test',name='test',user_data=None)


# facelist1 = CF.face_list.add_face(image='images/f2.jpg', face_list_id= 'test')

# print( facelist1)
# print(CF.face_list.lists())

match = CF.face.find_similars(faceid_webcam, face_list_id=None,
                              face_ids=[faceid_arash, faceid1, faceid2, faceid3, faceid4],
                              max_candidates_return=20,
                              mode='matchFace')


matchedFace = match[0]
matchedId = matchedFace['faceId']

if (matchedId == faceid_arash):
    print('face arash identified')
elif (matchedId == faceid2):
    print('face 2 identified')
elif (matchedId == faceid3):
    print('face 3 identified')
    # urllib2.urlopen("http://arash:12345@jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=1&switchcmd=On")
elif (matchedId == faceid4):
    print('face 4 identified')
elif (matchedId == faceid5):
    print('face 5 identified')
    cv2.imshow('image', face5)
else:
    print('no known face identified')

#
#
# # ## Helper functions
#
# # In[3]:
#
# def processRequest(json, data, headers, params):
#     """
#     Helper function to process the request to Project Oxford
#
#     Parameters:
#     json: Used when processing images from its URL. See API Documentation
#     data: Used when processing image read from disk. See API Documentation
#     headers: Used to pass the key information and the data type request
#     """
#
#     retries = 0
#     result = None
#
#     while True:
#
#         response = requests.request('post', _url, json=json, data=data, headers=headers, params=params)
#
#         if response.status_code == 429:
#
#             print("Message: %s" % (response.json()['error']['message']))
#
#             if retries <= _maxNumRetries:
#                 time.sleep(1)
#                 retries += 1
#                 continue
#             else:
#                 print('Error: failed after retrying!')
#                 break
#
#         elif response.status_code == 200 or response.status_code == 201:
#
#             if 'content-length' in response.headers and int(response.headers['content-length']) == 0:
#                 result = None
#             elif 'content-type' in response.headers and isinstance(response.headers['content-type'], str):
#                 if 'application/json' in response.headers['content-type'].lower():
#                     result = response.json() if response.content else None
#                 elif 'image' in response.headers['content-type'].lower():
#                     result = response.content
#         else:
#             print("Error code: %d" % (response.status_code))
#             print("Message: %s" % (response.json()['error']['message']))
#
#         break
#
#     return result
#
#
# # In[4]:
#
# def renderResultOnImage(result, img):
#     """Display the obtained results onto the input image"""
#
#     R = int(result['color']['accentColor'][:2], 16)
#     G = int(result['color']['accentColor'][2:4], 16)
#     B = int(result['color']['accentColor'][4:], 16)
#
#     cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color=(R, G, B), thickness=25)
#
#     if 'categories' in result:
#         categoryName = sorted(result['categories'], key=lambda x: x['score'])[0]['name']
#         cv2.putText(img, categoryName, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
#
# # # ## Analysis of an image retrieved via URL
# #
# # # In[5]:
# #
# # # URL direction to image
# # urlImage = 'https://oxfordportal.blob.core.windows.net/vision/Analysis/3.jpg'
# #
# # # Computer Vision parameters
# # params = {'visualFeatures': 'Color,Categories'}
# #
# # headers = dict()
# # headers['Ocp-Apim-Subscription-Key'] = _key
# # headers['Content-Type'] = 'application/json'
# #
# # json = {'url': urlImage}
# # data = None
# #
# # result = processRequest(json, data, headers, params)
# #
# # if result is not None:
# #     # Load the original image, fetched from the URL
# #     arr = np.asarray(bytearray(requests.get(urlImage).content), dtype=np.uint8)
# #     img = cv2.cvtColor(cv2.imdecode(arr, -1), cv2.COLOR_BGR2RGB)
# #
# #     renderResultOnImage(result, img)
# #
# #     ig, ax = plt.subplots(figsize=(15, 20))
# #     ax.imshow(img)
# #
# # ## Analysis of an image stored on disk
#
# # Load raw image file into memory
# pathToFileInDisk = r'images/f1.jpeg'
# with open(pathToFileInDisk, 'rb') as f:
#     data = f.read()
#
# # Computer Vision parameters
# params = {'visualFeatures': 'Color,Categories'}
#
# headers = dict()
# headers['Ocp-Apim-Subscription-Key'] = _key
# headers['Content-Type'] = 'application/octet-stream'
#
# json = None
#
# result = processRequest(json, data, headers, params)
#
# if result is not None:
#     # Load the original image, fetched from the URL
#     data8uint = np.fromstring(data, np.uint8)  # Convert string to an unsigned int array
#     img = cv2.cvtColor(cv2.imdecode(data8uint, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#
#     renderResultOnImage(result, img)
#
#     ig, ax = plt.subplots(figsize=(15, 20))
#     ax.imshow(img)
