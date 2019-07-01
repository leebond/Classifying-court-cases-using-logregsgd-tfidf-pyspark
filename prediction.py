# -*- coding: utf-8 -*-
"""
@author: Guanhua
"""
from __future__ import print_function
import sys
import re
import numpy as np
import sys
from pyspark.sql import Row
import pickle
from pyspark import SparkContext, SQLContext

ndim = 20000

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

def f1(df, cutoff):
    df = df.select("label", "pred", ((df.pred > cutoff).cast('float')).alias("pred_label"))
    df = df.select("label", "pred", "pred_label", (((df.label == 1.0) & (df.pred_label == 1.0)).cast('float')).alias("TP")
                   , (((df.label == 1.0) & (df.pred_label == 0.0)).cast('float')).alias("FN")\
                   , (((df.label == 0.0) & (df.pred_label == 0.0)).cast('float')).alias("TN")\
                   , (((df.label == 0.0) & (df.pred_label == 1.0)).cast('float')).alias("FP"))    
    sum_all = df.groupBy().sum().collect()
    TP = sum_all[0][0]
    FN = sum_all[0][1]
    TN = sum_all[0][2]
    FP = sum_all[0][3]
    prec = TP/(TP+FP)
    recall = TP/(TP+FN)
    F_measure = 2*prec*recall/(prec+recall)
    return F_measure
    
def best_f1(df, cutoffs):
    prev_f = 0
    for cutoff in cutoffs:
        print("cutoff: %f" %cutoff)
        df = df.select("label", "pred", ((df.pred > cutoff).cast('float')).alias("pred_label"))
        df = df.select("label", "pred", "pred_label", (((df.label == 1.0) & (df.pred_label == 1.0)).cast('float')).alias("TP")
                       , (((df.label == 1.0) & (df.pred_label == 0.0)).cast('float')).alias("FN")\
                       , (((df.label == 0.0) & (df.pred_label == 0.0)).cast('float')).alias("TN")\
                       , (((df.label == 0.0) & (df.pred_label == 1.0)).cast('float')).alias("FP"))
#            df.show()
        sum_all = df.groupBy().sum().collect()
        TP = sum_all[0][0]
        FN = sum_all[0][1]
        TN = sum_all[0][2]
        FP = sum_all[0][3]
        prec = TP/(TP+FP)
        recall = TP/(TP+FN)
        F_measure = 2*prec*recall/(prec+recall)
        print("F_measure: %f" %F_measure)
        if F_measure > prev_f:
            prev_f = F_measure
            best_cutoff = cutoff
    return F_measure, best_cutoff

if __name__ == "__main__":
    
    sc = SparkContext()
    sqlContext = SQLContext(sc)
    corpus = sc.textFile(sys.argv[1], 1)
    dictionary = sc.textFile(sys.argv[2], 1)
    with open('/tmp/weights.pkl','rb') as f:
        weights = pickle.load(f)[0]

    wordnRank = dictionary.map(lambda x: (x[x.index("'")+1: x.index(",")-1], int(x[x.index(",") + 2:][:-1] )) )

    validLines = corpus.filter(lambda x: 'id' in x and 'url=' in x)
   
    keyAndText = validLines.map(lambda x: (x[x.index('id="')+4: x.index('" url=')], x[x.index('">') + 2:][:-6] ))

    regex = re.compile('[^a-zA-Z]')
    keyAndListOfWords = keyAndText.map(lambda x: (str(x[0]), regex.sub(' ', x[1]).lower().split()) ).cache() # returns (id, [word1, word2, ...])
    allWords = keyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]) ) # returns (word, Docid) for all documents

    allDictionaryWords = allWords.join(wordnRank) # left join to dictionary; returns (word, (Docid, rank))

    justDocAndPos = allDictionaryWords.map(lambda x: (x[1][0], x[1][1])) # returns (Docid, rank) for all words

    allDictionaryWordsInEachDoc = justDocAndPos.groupByKey().mapValues(list) # returns (Docid, [rank, rank, ... for all words in document])

    features = allDictionaryWordsInEachDoc.map(lambda x: (x[0], buildArray(x[1]) ) ).cache()

    scores = features.map(lambda x: (x[0], computeTheta(x[1], weights)))

    prediction = scores.map(lambda x: (x[0], sigmoid(x[1])))


    prediction_df = prediction.map(lambda x: Row(label=float(1.0 if x[0][:2] == 'AU' else 0.0), pred=float(x[1])))
    df = sqlContext.createDataFrame(prediction_df).cache()
#    print(df.show())
    cutoff = 0.3
    F_measure = f1(df, cutoff)
#    print("Cutoff %f gives highest F1 score %f" %(best_cutoff, F_measure))
    print("Cutoff %f gives F1 score of %f" %(cutoff, F_measure))

    # save best F1 score for assignment
    myOutput = sc.parallelize([F_measure, cutoff], 1)
    myOutput.saveAsTextFile(sys.argv[3])

    sc.stop()