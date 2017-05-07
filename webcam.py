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


def match_face(webcam_image):
    KEY = '275f1ebbffeb49389decd1afb68e9157'  # Replace with a valid Subscription Key here.
    CF.Key.set(KEY)

    # img_url = 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'

    # result = CF.face.detect(img_url)
    # print(result)

    # pathToFileInDisk = r'images/f1.jpeg'
    # with open(pathToFileInDisk, 'rb') as f:
    #     im1 = f.read()
    result_webcam = CF.face.detect(webcam_image, landmarks=True, attributes='age,gender')

    result_arash = CF.face.detect('images/arash.jpg', landmarks=True, attributes='age,gender')

    result1 = CF.face.detect('images/f1.jpeg', landmarks=True, attributes='age,gender')
    result2 = CF.face.detect('images/f2.jpg', landmarks=True, attributes='age,gender')
    result3 = CF.face.detect('images/f3.jpg', landmarks=True, attributes='age,gender')
    result4 = CF.face.detect('images/f4.jpg', landmarks=True, attributes='age,gender')
    result5 = CF.face.detect('images/f5.jpg', landmarks=True, attributes='age,gender')

    print(result_webcam)

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

    result_face = ''

    if (matchedId == faceid_arash):
        result_face = ('face arash identified')
    elif (matchedId == faceid2):
        result_face = ('face 2 identified')
    elif (matchedId == faceid3):
        result_face = ('face 3 identified')
        # urllib2.urlopen("http://arash:12345@jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=1&switchcmd=On")
    elif (matchedId == faceid4):
        result_face = ('face 4 identified')
    elif (matchedId == faceid5):
        result_face = ('face 5 identified')
    else:
        result_face = ('no known face identified')

    return result_face


casc_path = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(casc_path)
face_detected = False
while not face_detected:
    video_capture = cv2.VideoCapture(0)

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    original_frame = frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces

    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        the_face = frame[y:y + h, x:x + w]
        cv2.imwrite('the_face.jpg', the_face)
        print("Face Detected")
        face_detected = True

video_capture.release()
cv2.destroyAllWindows()

print("processing face")
result_face = match_face('the_face.jpg')
print(result_face)


# cv2.imshow("cropped", theFace)

# Display the resulting frame


# cv2.imshow('Video', frame)


#
#



# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

# When everything is done, release the capture
