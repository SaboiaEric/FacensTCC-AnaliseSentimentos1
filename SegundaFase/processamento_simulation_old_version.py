from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from collections import Counter
import matplotlib.pyplot as plt
import itertools
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
    resultado = modelo.fit(treino_dados, treino_marcacoes)
    fit_and_predict_score = str(modelo.score(treino_dados, treino_marcacoes))
    print("ETAPA TREINO "+ nome + "- Acurácia: " + fit_and_predict_score)
    return fit_and_predict_score

def teste_real(modelo, validacao_dados, validacao_marcacoes):
    resultado = modelo.predict(validacao_dados)
    metricas(resultado,validacao_marcacoes)

def revalidacao_treino(modelo, treino_dados, treino_marcacoes):
    resultado = modelo.predict(treino_dados)

def metricas(resultado,validacao_marcacoes):
    #Apresentação do report contendo precision, recall and F-measures
    if len(set(resultado)) > 2:
        target_names = ['negativo', 'positivo', 'neutro']
        #os valores acima estão na ordem correta
        print(classification_report(validacao_marcacoes, resultado, target_names=target_names))
    else:
        print("Classification Report\n")
        target_names = ['negativo', 'positivo']
        #os valores acima estão na ordem correta
        print(classification_report(validacao_marcacoes, resultado, target_names=target_names))       

        #Apresentação dos true positive, false positive, true negative e false negative
        tn, fp, fn, tp = confusion_matrix(validacao_marcacoes, resultado).ravel() 
        msg = "True Positive: {0} \nFalse Positive: {1}\nTrue Negative: {2}\nFalse Negative: {3}".format(str(tn),str(fp),str(fn),str(tp))
        print(msg+"\n")

        matrix_confusao(validacao_marcacoes, resultado,target_names)

        print("Metrics\n")
        print("F-score: " + str(f1_score(validacao_marcacoes, resultado)))
        print("Precision: " + str(precision_score(validacao_marcacoes, resultado))) 
        print("Recall: " + str(recall_score(validacao_marcacoes, resultado))) 
        #Cria estatistica dos resultados.
        taxa_de_acerto_base = max(Counter(validacao_marcacoes).values()) * 100 / len(validacao_marcacoes)
        print("Taxa de acerto base: %f" % taxa_de_acerto_base)

    #Apresentação da accuracy e resultado final
    print("Acurácia: " + str(accuracy_score(validacao_marcacoes, resultado)))

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Matrix de Confusão Normalizada')
    else:
        print('Matrix de Confusão Não Normalizada')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def matrix_confusao(validacao_marcacoes, resultado, target_names):
    cnf_matrix = confusion_matrix(validacao_marcacoes, resultado)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=target_names,title='Matrix de Confusão, sem normalização')

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
    '''
    # OneVsRestClassifier - LinearSVC
    modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))
    resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes)
    resultados[resultadoOneVsRest] = modeloOneVsRest
    
    # OneVsOneClassifier
    modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state=0))
    resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes)
    resultados[resultadoOneVsOne] = modeloOneVsOne
    '''
    # MultinomialNB
    modeloMultinomial = MultinomialNB(alpha=1.0, fit_prior=True)
    resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
    resultados[resultadoMultinomial] = modeloMultinomial
    
    # AdaBoostClassifier
    modeloAdaBoost = AdaBoostClassifier(random_state=0)
    resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)
    resultados[resultadoAdaBoost] = modeloAdaBoost
    
    modeloDecisionTree = DecisionTreeClassifier(random_state=0)
    resultadoDecisionTree = fit_and_predict("DecisionTree", modeloDecisionTree, treino_dados, treino_marcacoes)
    resultados[resultadoDecisionTree] = modeloDecisionTree

    modeloRandomForest = RandomForestClassifier(random_state=0)
    resultadoRandomForest = fit_and_predict("RandomForest", modeloRandomForest, treino_dados, treino_marcacoes)
    resultados[resultadoRandomForest] = modeloRandomForest

    modeloLogisticRegression = LogisticRegression(random_state = 0)
    resultadoLogisticRegression = fit_and_predict("LogisticRegression", modeloLogisticRegression, treino_dados, treino_marcacoes)
    resultados[resultadoLogisticRegression] = modeloLogisticRegression
    
    #Verifica qual modelo teve a melhor perfomance
    maximo = max(resultados)
    vencedor = resultados[maximo]
    
    print("Vencedor: " + str(vencedor)+"\n")
    
    #Aplica o modelo vencedor em um caso do mundo real, utilizando dados nunca vistos
    vencedor.fit(treino_dados, treino_marcacoes)
    #revalidacao_treino(vencedor, treino_dados, treino_marcacoes)
    teste_real(vencedor, validacao_dados, validacao_marcacoes)
