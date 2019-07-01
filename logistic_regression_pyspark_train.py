# -*- coding: utf-8 -*-
"""
@author: Guanhua
"""
from __future__ import print_function
import sys
import re
import numpy as np
from operator import add
#from nltk.corpus import stopwords
from pyspark import SparkContext
import pickle
import gc

ndim = 20000
regParam = 0.5
learningRate = 0.05
maxiters = 100
precision = 0.01

def stringVector(x):
    returnVal = [np.float(x) for x in x[0]]
    return returnVal

def buildArray(listOfIndices):
    returnVal = np.zeros(ndim)
    for i in listOfIndices:
        if i != 20000:
            returnVal[i] = returnVal[i] + 1
        else:
            returnVal[i] = 0
    mysum = np.sum(returnVal) # rowsum
    returnVal = np.divide(returnVal, mysum)
    return returnVal

def computeTheta(arr, weights):
    return np.dot(weights, arr)

def sigmoid(myarr):
    return 1.0/(1 + np.exp(-myarr))

if __name__ == "__main__":
        
    sc = SparkContext()
    
    corpus = sc.textFile(sys.argv[1], 1)
    dictionary = sc.textFile(sys.argv[2], 1)
    

    wordnRank = dictionary.map(lambda x: (x[x.index("'")+1: x.index(",")-1], int(x[x.index(",") + 2:][:-1] )) )

    validLines = corpus.filter(lambda x: 'id' in x and 'url=' in x)

    keyAndText = validLines.map(lambda x: (x[x.index('id="')+4: x.index('" url=')], x[x.index('">') + 2:][:-6] ))

    regex = re.compile('[^a-zA-Z]')
    keyAndListOfWords = keyAndText.map(lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split()) ) # returns (id, [word1, word2, ...])
    allWords = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]) ) # returns (word, Docid) for all documents

    allDictionaryWords = allWords.join(wordnRank) # left join to dictionary; returns (word, (Docid, rank))

    justDocAndPos = allDictionaryWords.map(lambda x: (x[1][0], x[1][1])) # returns (Docid, rank) for all words

    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey().mapValues(list) # returns (Docid, [rank, rank, ... for all words in document])

    features = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1]) ) ).cache()

    i = 0
    numberOfDocs = validLines.count()
    weights = np.full((1, ndim) , 0.0)

    
    while i < maxiters:
        gc.collect()
        scores = features.map(lambda x: (x[0], computeTheta(x[1], weights)))

        prediction = scores.map(lambda x: (x[0], sigmoid(x[1])))

        target = prediction.map(lambda x: (x[0], 1.0 if x[0][:2] == 'AU' else 0.0 ))

        loss = target.join(prediction)

        computeLoss = loss.map(lambda x: (x[0], np.float(x[1][0]) - np.float(x[1][1])) )

        gradient = computeLoss.join(features)

        gradientVal = gradient.map(lambda x: (x[0], x[1][0]*x[1][1]) )

        gradientVal = gradientVal.reduce(lambda x, y: ("", np.add(x[1], y[1]) ))

        # Implement mean across all records
        weights += learningRate*(gradientVal[1]/np.float(numberOfDocs)) # - regParam*weights

        # TODO regularization
#        computeLossSq = computeLoss.map(lambda x: (x[0], np.power(x[1],2)) )
#        avgLoss = computeLossSq.reduce(lambda x, y: ("", np.add(x[1], y[1]) ))
#        print(avgLoss)
#        avgLossVal = avgLoss[1]/np.float(numberOfDocs)
#        print(avgLossVal)
        if i % 10 == 0:
            print("Iteration %d has completed." %i)
        i+=1    
    
    ind = np.argpartition(weights[0], -5)[-5:]

    mylist = []
    data = wordnRank.collect()
    for i in ind:
         mylist.append(data[i][0])
    
    myOutput = sc.parallelize(mylist, 1)
    myOutput.saveAsTextFile(sys.argv[3])
    
    with open('/tmp/weights.pkl','wb') as f:
        pickle.dump(weights, f)
    
    sc.stop()
    