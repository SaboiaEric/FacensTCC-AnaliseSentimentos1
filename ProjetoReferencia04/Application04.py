import json
from textblob import TextBlob as tb
import tweepy
import numpy as np 

#Let's go to the code :)


# Load credentials from json file
with open("twitter_credentials.json", "r") as file:  
    creds = json.load(file)

#Twitter Authentication
auth = tweepy.OAuthHandler(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
auth.set_access_token(creds['ACCESS_TOKEN'], creds['ACCESS_SECRET'])
api = tweepy.API(auth)

#Twitter's search
public_tweets = api.search('Eleicao2018')

#Data
print(len(public_tweets))