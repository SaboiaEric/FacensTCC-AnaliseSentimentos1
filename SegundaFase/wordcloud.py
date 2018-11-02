import pandas as pd
import numpy as np
import nltk
import json
import pickle
import csv
from collections import Counter

tweets = pd.read_excel('DataSet_Bolsonaro.xlsx', sheet_name='DS-PN-ID', encoding='utf-8')
frases = tweets['Text'].str.lower()

print("Arquivo: DataSet_Bolsonaro.xlsx	-  Folha: DS-PN-ID")

textosQuebrados = [nltk.tokenize.word_tokenize(frase) for frase in frases]
stopwords = nltk.corpus.stopwords.words('portuguese')
stemmer = nltk.stem.RSLPStemmer();
'''    
dicionario = set()
for lista in textosQuebrados:
    validas = [stemmer.stem(palavra) for palavra in lista if palavra not in stopwords and len(palavra) > 2]
    dicionario.update(validas)
    

totalDePalavras = len(dicionario)
tuplas = zip(dicionario, range(totalDePalavras))
'''
palavras = []
for lista in textosQuebrados:
    for palavra in lista:
        if palavra not in stopwords and len(palavra) > 2:
    	    palavras.append(palavra)    


#print(palavras)
for elemento in palavras:
	print(elemento +": "+ str(palavras.count(elemento)))
	#print(palavras[elemento] +": "+ (Counter(palavras[elemento]).values()))


'''
dicionario = []
for lista in textosQuebrados:
    for palavra in lista: 
        if palavra not in stopwords and len(palavra) > 2:
            tuplas = zip(dicionario, range(totalDePalavras))


dicionario = []
for lista in textosQuebrados:
    print(lista)

totalDePalavras = len(dicionario)

print(tuplas)
dicionario = sorted(dicionario)

print(dicionario.count("bolsonaro"))
with open('your_file.txt', 'w', encoding='utf-8') as f:
    for item in dicionario:
        f.write(str(dicionario.count(item)) +" - " + item +"\n")

'''