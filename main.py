



from __future__ import print_function
import time
import requests
import cv2
import operator
import numpy as np
import cognitive_face as CF




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
result1 = CF.face.detect('images/f1.jpeg')
print(result1)
result2 = CF.face.detect('images/f2.jpg')
print(result2)
CF.face.find_similars()

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
