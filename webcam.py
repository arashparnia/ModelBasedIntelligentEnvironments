from __future__ import print_function

import calendar
import time
from pprint import pprint

import requests
import cv2
import operator
import numpy as np
import cognitive_face as CF

import json
import urllib2
import requests
from requests.auth import HTTPBasicAuth


# michel idx = 17 working stever
# aart idx = 16 chaotic chriss

#
# headers = {'Authorization': 'Basic QXJhc2g6MTIzNDU='}
#
# request = urllib2.Request("http://jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=16&switchcmd=On",
#                           headers=headers)
#
# contents = urllib2.urlopen(request).read()


current_sec = lambda: int(round(time.time() * 1000))


# import requests





def recognized_chaotic_chris_on():
    headers = {'Authorization': 'Basic QXJhc2g6MTIzNDU='}

    request = urllib2.Request(
        "http://jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=16&switchcmd=On",
        headers=headers)

    contents = urllib2.urlopen(request).read()


def recognized_chaotic_chris_off():
    headers = {'Authorization': 'Basic QXJhc2g6MTIzNDU='}

    request = urllib2.Request(
        "http://jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=16&switchcmd=Off",
        headers=headers)

    contents = urllib2.urlopen(request).read()


def recognized_working_steve_on():
    headers = {'Authorization': 'Basic QXJhc2g6MTIzNDU='}

    request = urllib2.Request(
        "http://jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=17&switchcmd=On",
        headers=headers)

    contents = urllib2.urlopen(request).read()


def recognized_working_steve_off():
    headers = {'Authorization': 'Basic QXJhc2g6MTIzNDU='}

    request = urllib2.Request(
        "http://jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=17&switchcmd=Off",
        headers=headers)

    contents = urllib2.urlopen(request).read()


recognized_chaotic_chris_off()
recognized_working_steve_off()


# direct telegram message
# https://api.telegram.org/bot353028451:AAEoXtjebnEbxWLJORYLGhTyie_co3u0HRo/sendmessage?chat_id=277127047&text=test

# print(urllib.urlopen('http://jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=1&switchcmd=On').read())
# urllib2.urlopen('http://arash:12345@jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=1&switchcmd=On')
# urllib2.urlopen('http://arash:12345@jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=1&switchcmd=Off')
# http://arash:12345@jebo.mynetgear.com:8080/json.htm?type=command&param=switchlight&idx=1&switchcmd=On




def match_face(webcam_image):
    KEY = '74d1bf4b5f194be38ddab0405a17f793 '  # Replace with a valid Subscription Key here.
    CF.Key.set(KEY)

    # img_url = 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'

    # result = CF.face.detect(img_url)
    # print(result)

    # pathToFileInDisk = r'images/f1.jpeg'
    # with open(pathToFileInDisk, 'rb') as f:
    #     im1 = f.read()

    # print(CF.face_list.create('face_list_1', name='face_list_1', user_data=None))

    # print(CF.face_list.add_face(image='aart.jpg', face_list_id='face_list_1'))

    result_webcam = CF.face.detect(webcam_image, landmarks=False, attributes='age,gender')


    print(result_webcam)

    face_webcam = result_webcam[0]

    faceid_webcam = (face_webcam['faceId'])

    match = CF.face.find_similars(faceid_webcam, face_list_id='face_list_1',
                                  face_ids=None,
                                  max_candidates_return=20,
                                  mode='matchFace')

    matchedFace = match[0]
    print(match)
    matchedId = matchedFace['persistedFaceId']
    confidence = matchedFace['confidence']

    result_face = 'no known faces'
    if confidence > 0.5:
        if (matchedId == '507f97f8-e5b7-4bb9-ad85-d497e8505eb3'):
            result_face = ('arash identified')
        elif (matchedId == '3d554532-d389-47f8-890b-6ff343058285'):
            result_face = ('michel identified')
            recognized_working_steve_on()
        elif (matchedId == 'f09299d7-518d-4390-9245-7794933960f2'):
            result_face = ('aart identified')
            recognized_chaotic_chris_on()
            # elif (matchedId == faceid5):
            #     result_face = ('face 5 identified')

    return result_face


casc_path = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(casc_path)
face_detected = False
# video_capture.release()
# cv2.destroyAllWindows()

while not face_detected:
    # while True:
    video_capture = cv2.VideoCapture(0)

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # recognizing face

    # print(current_sec() % 10)
    # if current_sec() % 10 == 0:
    for (x, y, w, h) in faces:
        # the_face = frame[y:y + h, x:x + w]
        cv2.imwrite('the_face.jpg', frame)
        print("Face Detected")
        face_detected = True

    cv2.imshow("frame", frame)




    # Draw a rectangle around the faces
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #

    # cv2.imshow("frame", frame)

    import threading

    # def ():
    #     threading.Timer(5.0, printit).start()
    #     print
    #     "Hello, World!"
    #
    #
    # printit()


    # recognized_chaotic_chris_off()
    # recognized_working_steve_off()



    # time.sleep(15)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


video_capture.release()
cv2.destroyAllWindows()

print("processing face")
result_face = match_face('the_face.jpg')
print(result_face)



# Display the resulting frame


# cv2.imshow('Video', frame)


#
#



# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

# When everything is done, release the capture
