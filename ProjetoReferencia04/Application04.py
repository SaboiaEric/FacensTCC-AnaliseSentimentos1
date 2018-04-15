import json
from textblob import TextBlob as tb
import tweepy
import numpy as np 
import datetime
import re

#Tweet structure
def process_tweet(tweet):  
    d = {}
    #d['hashtags'] = [hashtag['text'] for hashtag in tweet['entities']['hashtags']]
    d['id_str'] = tweet.id_str
    d['created_at'] = tweet.created_at
    d['text'] = tweet.text
    d['user_id_str'] = tweet.user.id_str
    d['user_name'] = tweet.user.name
    return d

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

#Creating safe tweet
def create_string(item):
    obj = "ID:" +  item['id_str'] + " - "
    obj += "Criado em:" + str(item['created_at']) + " - " 
    obj += "Texto:" + item['text'] + " - " 
    obj += "Usuario ID:" + item['user_id_str'] + " - " 
    obj += "Usuario Nome:"+ item['user_name'] + "\n"
    return obj

#Saving tweet
def save_to_txt(tweets):
    currentlyTime = str(datetime.datetime.now()).replace(':','_').replace(' ','_').replace('-','_')
    with open(currentlyTime + "-" + "miniDatabase.txt", "w") as file:
        for item in tweets:
            tweet_safe = remove_emoji(create_string(item))
            file.write(tweet_safe)
        
        #writer.writerow(list(tweets.values()))

# Load credentials from json file
with open("twitter_credentials.json", "r") as file:  
    creds = json.load(file)

#Twitter Authentication
auth = tweepy.OAuthHandler(creds['CONSUMER_KEY'], creds['CONSUMER_SECRET'])
auth.set_access_token(creds['ACCESS_TOKEN'], creds['ACCESS_SECRET'])
api = tweepy.API(auth)

#Twitter's search
public_tweets = api.search('Eleicao2018')

tweets = []
for tweet in public_tweets:
    try:
        tw = process_tweet(tweet)
        tweets.append(tw)
    except:
        print("ERROR TRY AGAIN\n")
#print(list(tweets))
save_to_txt(tweets)

#Data
print(len(public_tweets))