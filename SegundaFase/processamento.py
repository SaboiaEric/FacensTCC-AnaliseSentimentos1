from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import average_precision_score ,accuracy_score ,confusion_matrix, classification_report
from collections import Counter
import pandas as pd
import numpy as np
import nltk
import json
import pickle
import csv

def vetorizar_texto(texto, tradutor, stemmer):
    vetor = [0] * len(tradutor)
    for palavra in texto:
        if len(palavra) > 0:
            raiz = stemmer.stem(palavra)
            if raiz in tradutor:
                posicao = tradutor[raiz]
                vetor[posicao] += 1
    return vetor

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
    k = 10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv=k)
    taxa_de_acerto = np.mean(scores)
    msg = "ETAPA TREINO - Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
    print(msg)
    return taxa_de_acerto

def teste_real(modelo, validacao_dados, validacao_marcacoes):
    resultado = modelo.predict(validacao_dados)
    metricas(resultado,validacao_marcacoes)

def metricas(resultado,validacao_marcacoes):
    #Apresentação do report contendo precision, recall and F-measures
    if len(set(resultado)) > 2:
        target_names = ['negativo', 'positivo', 'neutro']
        #os valores acima estão na ordem correta
        print(classification_report(validacao_marcacoes, resultado, target_names=target_names))
    else:
        target_names = ['negativo', 'positivo']
        #os valores acima estão na ordem correta
        print(classification_report(validacao_marcacoes, resultado, target_names=target_names))
        
        #Apresentação dos true positive, false positive, true negative e false negative
        tn, fp, fn, tp = confusion_matrix(validacao_marcacoes, resultado).ravel() 
        msg = "true positive: {0} - false positive: {1} - true negative: {2} - false negative: {3}".format(str(tn),str(fp),str(fn),str(tp))
        print(msg)

    #Apresentação da accuracy e resultado final
    msg = "ETAPA VALIDAÇÃO - Taxa de acerto com dados do mundo real: " + str(accuracy_score(validacao_marcacoes, resultado))
    print(msg)

def processar(tweets, frases):
    #Descomente caso seja primeira vez de execução. Funcionalidaeds de processamento
    #nltk.download('stopwords')
    #nltk.download('rslp')
    #nltk.download('punkt')

    #aplica a tokenização e cria a back of words
    textosQuebrados = [nltk.tokenize.word_tokenize(frase) for frase in frases]

    #aplica as stop words, isto é, remove os pronomes e etc.
    stopwords = nltk.corpus.stopwords.words('portuguese')
    
    #aplica steem, isto é, deixa a palavra na forma raiz
    stemmer = nltk.stem.RSLPStemmer();
    
    dicionario = set()
    for lista in textosQuebrados:
        validas = [stemmer.stem(palavra) for palavra in lista if palavra not in stopwords and len(palavra) > 2]
        dicionario.update(validas)
    #print(dicionario)


    totalDePalavras = len(dicionario)
    #print(totalDePalavras)
    #print(dicionario)
    tuplas = zip(dicionario, range(totalDePalavras))
    tradutor = {palavra: indice for palavra, indice in tuplas}


    vetoresDeTexto = [vetorizar_texto(texto, tradutor, stemmer) for texto in textosQuebrados]
    
    marcas = tweets['Sentimento']
    
    X = vetoresDeTexto
    Y = marcas
    
    #Divisão de dados para treino, teste e validação.
    porcentagem_de_treino = 0.8
    
    tamanho_de_treino = int(porcentagem_de_treino * len(Y))
    tamanho_de_validacao = len(Y) - tamanho_de_treino
    
    treino_dados = X[0:tamanho_de_treino]
    treino_marcacoes = Y[0:tamanho_de_treino]
    
    validacao_dados = X[tamanho_de_treino:]
    validacao_marcacoes = Y[tamanho_de_treino:]
    
    
    
    resultados = {}
    
    # OneVsRestClassifier - LinearSVC
    modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))
    resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes)
    resultados[resultadoOneVsRest] = modeloOneVsRest
    
    # OneVsOneClassifier
    modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state=0))
    resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes)
    resultados[resultadoOneVsOne] = modeloOneVsOne
    
    # MultinomialNB
    modeloMultinomial = MultinomialNB(alpha=1.0, fit_prior=True)
    resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
    resultados[resultadoMultinomial] = modeloMultinomial
    
    # AdaBoostClassifier
    modeloAdaBoost = AdaBoostClassifier(random_state=0)
    resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)
    resultados[resultadoAdaBoost] = modeloAdaBoost
    
    #Verifica qual modelo teve a melhor perfomance
    maximo = max(resultados)
    vencedor = resultados[maximo]
    
    print("Vencedor: " + str(vencedor))
    
    #Aplica o modelo vencedor em um caso do mundo real, utilizando dados nunca vistos
    vencedor.fit(treino_dados, treino_marcacoes)
    teste_real(vencedor, validacao_dados, validacao_marcacoes)
    
    #Cria estatistica dos resultados.
    acerto_base = max(Counter(validacao_marcacoes).values())
    taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
    print("Taxa de acerto base: %f" % taxa_de_acerto_base)
    
    total_de_elementos = len(validacao_dados)
    print("Total de teste: %d" % total_de_elementos)

