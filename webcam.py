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
    KEY = '74d1bf4b5f194be38ddab0405a17f793 '  # Replace with a valid Subscription Key here.
    CF.Key.set(KEY)

    # img_url = 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'

    # result = CF.face.detect(img_url)
    # print(result)

    # pathToFileInDisk = r'images/f1.jpeg'
    # with open(pathToFileInDisk, 'rb') as f:
    #     im1 = f.read()

    # print(CF.face_list.create('face_list_1', name='face_list_1', user_data=None))

    # print(CF.face_list.add_face(image='images/', face_list_id='face_list_1'))

    result_webcam = CF.face.detect(webcam_image, landmarks=False, attributes='age,gender')

    # result_arash = CF.face.detect('images/arash.jpg', attributes='age,gender')
    #
    # result1 = CF.face.detect('images/f1.jpeg', attributes='age,gender')
    # result2 = CF.face.detect('images/f2.jpg', attributes='age,gender')
    # result3 = CF.face.detect('images/f3.jpg', attributes='age,gender')
    # result4 = CF.face.detect('images/f4.jpg', attributes='age,gender')
    # result5 = CF.face.detect('images/f5.jpg', attributes='age,gender')

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
            # elif (matchedId == faceid4):
            #     result_face = ('face 4 identified')
            # elif (matchedId == faceid5):
            #     result_face = ('face 5 identified')

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
