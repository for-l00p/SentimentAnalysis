__author__ = 'Shubham'
import tweepy
import re
import pprint
auth = tweepy.OAuthHandler("VsYLafT54hieigNEGiuIdW36Q", "oTEWT3SDbPD7HAUAp2LLdWtknzqlj0pv7TS4rbVn5SvhvCf7aX")
auth.set_access_token("88892407-11MKyMvMBP33rHweUHnCDKzDi3z5fl2xpvktVePeD", "Lafadj3oRr6TVQKrWEEakZXko5H45EGZFxDNyzFYzrq0d")

api = tweepy.API(auth)

#public_tweetsapi.trends_available
'''
public_tweets = api.trends_place(20070458)
x=""
for tweet in public_tweets:
    x+=str(tweet)
    pprint.pprint(tweet)
'''
x=""
import json
c = tweepy.Cursor(api.search, q='goa',lang='en')
for tweet in c.items():
    #data = json.load(tweet)
    #try:
    #print tweet
    try:
        print tweet.text
    except:
        pass

    x+=str( tweet)
    #print data["text"]
#print x
'''
p= x.decode("windows-1252")
x = p.encode("utf8")
regex="text=u'(.+?)'"
pattern=re.compile(regex)
y=re.findall(pattern,x)
j=0
while j<len(y):
    print y[j]
    j+=1
'''

