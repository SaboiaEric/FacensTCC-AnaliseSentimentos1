from textblob import TextBlob as tb
import tweepy
#Importado numpy para medir a popularidade dos modos
#Não é necessário que isso seja feito :)
import numpy as np 

#Let's go to the code :)

#for item in tweets:
consumer_key = 'KJUcp9nAVbqjzrTgoMnX3huIy'
consumer_secret = 'd9IB0sm8EXacPY6KcBxSJ2e3ULghvx4Y7zBRQQRKPaY4tS3OO7'

access_token = '839943593322217473-7b2CeeFLaJFNMf7KUxMk6boKlVhemPK'
access_token_secret = 'AOnWdLm8iKS1QunnsUmOq1zZCWAAff5YR49EzMTAPM6e4'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

#Variável que irá armazenar todos os Tweets com a palavra escolhida na função search da API
public_tweets = api.search('Eleicao2018')
#Variável que irá armazenar as polaridades
analysis = None

#Declarando a lista contendo as informações dos tweets
tweets = []
for tweet in public_tweets:
    try:
        #print(tweet.text)
        analysis = tb(tweet.text)
        #print(analysis.sentiment.polarity)
        #print('MÉDIA DE SENTIMENTO: ' + str(np.mean(analysis.sentiment.polarity)))
        tweets.append(tweet)
    except:
        continue

print(len(tweets))
print("\n\n\n\n")
#lista_completa = ''.join(tweets)

#for item in tweets:
    #print(str(item))
    #print("\n")
    #print(item.text +"\n")

import datetime

currentlyTime = str(datetime.datetime.now()).replace(':','_').replace(' ','_').replace('-','_')
with open(currentlyTime + " - " + "miniDatabase.txt", "w") as output:
    for tweet in tweets:
        output.write(str(tweet.id)+ " -- " + tweet.text + "\n")