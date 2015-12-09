#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    01/10/2015
#
# *************************************** #

from tweepy import Stream
import tweepy, argparse, datetime
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from twitter_keys import consumer_key, consumer_secret, access_token, access_secret
import time, os, io
import sys

write_pattern = '{0}\n'

class MyListener(StreamListener):
    def on_status(self, status):
        # print status.text
        with open(key + '.csv', 'a') as f:
                f.write(status.text.encode('utf-8'))
                f.write('\n')

    # def on_data(self, data):
    #     try:
    #         print("Data ...")
    #         print data
    #         exit(0)
    #         with open('test.csv', 'a') as f:
    #             data=data.encode('utf-8')
    #             f.write(data)
    #         return True
    #     except BaseException as e:
    #         print("Error on_data: %s" % str(e))
    #         return True

    def on_error(self, status):
        print(status)
        return True
"""
parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', required=True, help='The name of the file')
parser.add_argument('--key', '-k', required=True, help='The key word')
args = parser.parse_args()
log = args.file
key = args.key #??? ????? ???? ???????? ? ??????-????????

try:
    fo = open(log, "wb")
    print "Name of the file: ", fo.name
    print "Opening mode : ", fo.mode
    fo.write(write_pattern.format(datetime.date.today().strftime('%d %b %Y')))
    fo.write(write_pattern.format('The key word is: {0}'.format(args.key)))
    fo.close()
except Exception, e:
    print 'An Error occurred:\n{0}'.format(str(e))
"""

parser = argparse.ArgumentParser()
# parser.add_argument('--file', '-f', required=True, help='The name of the file')
parser.add_argument('--key', '-k', required=True, help='The key word')
args = parser.parse_args()
# log = args.file
key = args.key #??? ????? ???? ???????? ? ??????-????????
# print key, type(key), key.decode('utf-8')

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)
file = open(key + '.csv', 'wb')
file.close()
twitter_stream = Stream(auth, MyListener())
twitter_stream.filter(track=[key.decode('utf-8')])
#twitter_stream.filter(track=[key])