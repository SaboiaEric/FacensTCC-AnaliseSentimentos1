import pandas as pd
import numpy as np
import nltk
import json
import pickle
import csv

tweets = pd.read_excel('DataSet_Bolsonaro.xlsx', sheet_name='DS-PN-ID', encoding='utf-8')
frases = tweets['Text'].str.lower()

print("Arquivo: DataSet_Bolsonaro.xlsx	-  Folha: DS-PN-ID")

textosQuebrados = [nltk.tokenize.word_tokenize(frase) for frase in frases]

#aplica as stop words, isto é, remove os pronomes e etc.
stopwords = nltk.corpus.stopwords.words('portuguese')

'''
lista_palavras = [] 
for lista in textosQuebrados:
    	lista_palavras.append(lista)
'''
#aplica steem, isto é, deixa a palavra na forma raiz
stemmer = nltk.stem.RSLPStemmer();
    
    
dicionario = []
for lista in textosQuebrados:
    for palavra in lista: 
    	if palavra not in stopwords and len(palavra) > 2:
    		dicionario.append(palavra)

'''totalDePalavras = len(dicionario)

print(tuplas)
dicionario = sorted(dicionario)'''

print(dicionario.count("bolsonaro"))
with open('your_file.txt', 'w', encoding='utf-8') as f:
    for item in dicionario:
        f.write(str(dicionario.count(item)) +" - " + item +"\n")