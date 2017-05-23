import urllib2
from pandas import json

from sklearn import preprocessing
import numpy
import pandas
import random
import bisect
import collections
import numpy as np


def read_status_for(id):
    headers = {'Authorization': 'Basic QXJhc2g6MTIzNDU='}

    request = urllib2.Request(
        "http://jebo.mynetgear.com:8080/json.htm?type=devices&rid=" + str(id),
        headers=headers)

    contents = urllib2.urlopen(request).read()
    r = json.loads(contents)
    r = r["result"]
    r = r[0]
    r = r['Status']
    if (r == 'On'):
        return (1)
    elif (r == 'Off'):
        return (0)


def telegram_message_chaitic_chris(m):
    headers = {'Authorization': ' a'}
    # working steve
    # request_text = "https://api.telegram.org/bot349402760:AAHCI67j_KP4yL8NFB83gAvNQkxMFdPQHgw/sendmessage?chat_id=354794269&text=" + str(q)
    # chaitic chris
    request_text = "https://api.telegram.org/bot354053678:AAFlWOgzArXbF3JaQ0oZdIHxX0bruXQzdMY/sendmessage?chat_id=376206693&text=" + str(
        m)

    request = urllib2.Request(request_text, headers=headers)

    contents = urllib2.urlopen(request).read()


def telegram_message_working_steve(m):
    headers = {'Authorization': ' a'}
    # working steve
    request_text = "https://api.telegram.org/bot349402760:AAHCI67j_KP4yL8NFB83gAvNQkxMFdPQHgw/sendmessage?chat_id=354794269&text=" + str(
        m)
    # chaitic chris
    # request_text = "https://api.telegram.org/bot354053678:AAFlWOgzArXbF3JaQ0oZdIHxX0bruXQzdMY/sendmessage?chat_id=376206693&text=" + str(m)

    request = urllib2.Request(request_text, headers=headers)

    contents = urllib2.urlopen(request).read()


unknown_presence = read_status_for(18)

Chaotic_Chris_presence = read_status_for(14)
Chaotic_Chris_camera = read_status_for(16)

Working_Steve_presence = read_status_for(15)
Working_Steve_camera = read_status_for(17)

door = read_status_for(10)

print(door, Chaotic_Chris_presence, Chaotic_Chris_camera)
print (door, Working_Steve_presence, Working_Steve_camera)

# current_status_chaiotic_chris = [Chaotic_Chris_camera,Chaotic_Chris_presence,door]
#
#
# print("current status " , current_status_chaiotic_chris)




p1111 = [
    6,
    11,
    11,
    8,
    7]

p1112 = [
    13,
    12,
    12,
    10,
    12]

p1113 = [
    13,
    13,
    11,
    13,
    16]

p1114 = [
    14,
    17,
    18,
    14,
    14]

p1115 = [
    24,
    20,
    28,
    30,
    22
]

q1111 = [
    12,
    12,
    12,
    13,
    11]
q1112 = [
    14,
    12,
    13,
    14,
    12]
q1113 = [
    15,
    18,
    14,
    17,
    14]
q1114 = [
    18,
    18,
    24,
    22,
    21]
q1115 = [
    36,
    30,
    24,
    21,
    21]

sum_of_p = (numpy.average(p1111)) + (numpy.average(p1112)) + (numpy.average(p1113)) + (numpy.average(p1114)) + (
numpy.average(p1115))

p = [1 - (numpy.average(p1111)) / sum_of_p, 1 - (numpy.average(p1112)) / sum_of_p,
     1 - (numpy.average(p1113)) / sum_of_p, 1 - (numpy.average(p1114)) / sum_of_p,
     1 - (numpy.average(p1115)) / sum_of_p]
# print(p)
# print (sum(p))
sum_of_q = (numpy.average(q1111)) + (numpy.average(q1112)) + (numpy.average(q1113)) + (numpy.average(q1114)) + (
numpy.average(q1115))

q = [1 - (numpy.average(q1111)) / sum_of_q, 1 - (numpy.average(q1112)) / sum_of_q,
     1 - (numpy.average(q1113)) / sum_of_q, 1 - (numpy.average(q1114)) / sum_of_q,
     1 - (numpy.average(q1115)) / sum_of_q]

# pn = preprocessing.normalize(p, axis=1, norm='l2', copy=True)
# pn = pn[0]
# print (pn)



#

chaiotic_chris_warning_message = ['Chris, if you close the door now within 8 seconds, you will be faster than Steve!',
                                  'Hey Chris! Close the door, you do not want to be slower than your grandma?',
                                  'Hey Chris! You would be awesome if you could close the door, it will save some energy!',
                                  'Can you please close the door Chris?',
                                  'Close the door Chris!'
                                  ]

chaiotic_chris_thank_you_message = [
    'Thank you for closing the door, you will become the fastest family member if you keep this up!',
    'At least you are better than Feyenoord and closed the door behind you!',
    'Thanks Chris! You are the best!',
    'Thank you for closing the door, Chris!',
    'Thank you'
    ]

working_steve_warning_message = [
    'Can you close the door again Steve? By closing the door you will even save more money!',
    'Can you close the door, Steve? You are on your way to save more energy than you did last month!',
    'If you close the door, you will do better than your neighbours!',
    'Can you please close the door, Steve?',
    'Close the door Steve!'
    ]

working_steve_thank_you_message = [
    'Thank you for closing the door, you will become the fastest family member if you keep this up!',
    'At least you are better than Feyenoord and closed the door behind you!',
    'Thanks Chris! You are the best!',
    'Thank you for closing the door, Chris!',
    'Thank you'
    ]

if (door == 1 and Chaotic_Chris_presence == 1 and Chaotic_Chris_camera == 1):
    m = np.random.choice(chaiotic_chris_warning_message, 1, p)
    print ("chaiotic_chris_warning_message", m)
    telegram_message_chaitic_chris(m)

if (door == 1 and Working_Steve_presence == 1 and Working_Steve_camera == 1):
    m = np.random.choice(working_steve_warning_message, 1, p)
    print ("working_steve_warning_message", m)
    telegram_message_working_steve(m)

if door == 0 and Chaotic_Chris_presence == 1 and Chaotic_Chris_camera == 1:
    m = np.random.choice(chaiotic_chris_thank_you_message, 1, p)
    print ("thank chaiotic_chris_thank_you_message message", m)
    telegram_message_chaitic_chris(m)

if (door == 0 and Working_Steve_presence == 1 and Working_Steve_camera == 1):
    m = np.random.choice(working_steve_thank_you_message, 1, p)
    print ("working_steve_thank_you_message", m)
    telegram_message_working_steve(m)

if (door == 0 and Working_Steve_presence == 0 and Working_Steve_camera == 0
    and Chaotic_Chris_presence == 0 and Chaotic_Chris_camera == 0 and unknown_presence == 0
    ):
    m = "DOOR CLOSED NO PRESENCE DETECTED"
    print (m)
    telegram_message_chaitic_chris(m)
    telegram_message_working_steve

unknown_presence

if (unknown_presence == 1):
    m = "WARNING! UNKNOWN PRESENCE DETECTED"
    print (m)
    telegram_message_chaitic_chris(m)
    telegram_message_working_steve
