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
tweets = api.search('Eleicao2018')
#Variável que irá armazenar as polaridades

for tweet in tweets:
    frase = tb(tweet.text)

    #Detecta a linguagem e se necessário converte para inglês
    if frase.detect_language() != 'en':
        traducao = tb(str(frase.translate(to='en')))
        print('Tweet: {0} - Sentimento: {1}'.format(tweet.text, traducao.sentiment))
    else:
        print('Tweet: {0} - Sentimento: {1}'.format(tweet.text, frase.sentiment))

print(len(tweets))