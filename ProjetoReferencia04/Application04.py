import json
from textblob import TextBlob as tb
import tweepy
import numpy as np 
import datetime

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

#Saving tweet
def save_to_txt(tweets):
    currentlyTime = str(datetime.datetime.now()).replace(':','_').replace(' ','_').replace('-','_')
    with open(currentlyTime + "-" + "miniDatabase.txt", "w") as file:
        for item in tweets:
            obj = item['id_str'] + "-" + str(item['created_at']) + "-" + item['text'] + "-" + item['user_id_str'] + "-" + item['user_name']
            file.write(str(obj.encode('utf-8')))
        
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

print("processou tudo")
#print(list(tweets))
save_to_txt(tweets)

#Data
print(len(public_tweets))