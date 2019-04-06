import os
import pickle
import tqdm
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from itertools import groupby
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from collections import Counter


def calMean(gapList):
    errorValues = np.mean(gapList)
    return errorValues
    # print("Error values: ",errorValues)

def ts(predicted,labels,threshold):
    tp = np.sum((predicted >= threshold) * (labels >= threshold))
    tn = np.sum((predicted < threshold) * (labels < threshold))
    fp = np.sum((predicted < threshold) * (labels >= threshold))
    fn = np.sum((predicted >= threshold) * (labels < threshold))

    ts = round(tp / (tp + fp + fn), 5)
    return ts


def calPmae(predicted,labels):
    gaps = np.abs(predicted-labels)
    pCount = np.sum(labels>0.05)
    pError = np.sum((labels>0.05)*gaps)
    mpae = round(pError/pCount,5)
    return mpae


def benchmarkGenerator():
    """
    :param yearMonth:
    :return:
    """
    cropRoot = "/mnt/pami/xyxu/dataset/cropGrib"
    random.seed(0)
    totalGribFileList = os.listdir(cropRoot)
    random.shuffle(totalGribFileList)
    cropFileList = [os.path.join(cropRoot, item) for item in
                    totalGribFileList[:int(0.8 * len(totalGribFileList))]]
    testCropFileList = [os.path.join(cropRoot, item) for item in
                        totalGribFileList[-int(0.2 * len(totalGribFileList)):]]




    res = []

    for fileName in tqdm.tqdm(cropFileList[:], total=len(cropFileList),
                              desc="FileName", ncols=80, leave=False):
        with open(fileName, 'rb') as f:
            oneCropDataDict = pickle.load(f)
        # print(oneCropDataDict)
        for cropDataArray in tqdm.tqdm(oneCropDataDict.values(), total=len(oneCropDataDict),
                                       desc="One File Data", ncols=80, leave=False):
            precipitationArray = cropDataArray.loc['Total precipitation', :, :]
            micapsValue = cropDataArray.attrs["micapValues"]
            gribValue = np.mean(precipitationArray.values[8:10, 8:10]) * 1000
            gapValue = gribValue - micapsValue
            if micapsValue>=0.1:
                res.append([micapsValue])

    cls = KMeans(n_clusters=3, random_state=0).fit(res)
    print(cls.cluster_centers_)

    totalCount = 0
    zeroCount = 0
    test = []
    labels = []
    for fileName in tqdm.tqdm(testCropFileList[:], total=len(testCropFileList),
                              desc="FileName", ncols=80, leave=False):
        with open(fileName, 'rb') as f:
            oneCropDataDict = pickle.load(f)
        # print(oneCropDataDict)
        for cropDataArray in tqdm.tqdm(oneCropDataDict.values(), total=len(oneCropDataDict),
                                       desc="One File Data", ncols=80, leave=False):
            precipitationArray = cropDataArray.loc['Total precipitation', :, :]
            gribValue = np.mean(precipitationArray.values[8:10, 8:10]) * 1000
            micapsValue = cropDataArray.attrs["micapValues"]
            if gribValue>=0.1:
                totalCount += 1
                test.append([gribValue])
                labels.append([micapsValue])

    res1 = cls.predict(test)
    res2 = cls.predict(labels)

    correct = np.sum(res1==res2)
    print(correct)
    print(totalCount)
    print(round(correct/totalCount,5))

def rangeStatistics():
    cropRoot = "/mnt/pami/xyxu/dataset/cropGrib"
    random.seed(0)
    totalGribFileList = os.listdir(cropRoot)
    random.shuffle(totalGribFileList)
    cropFileList = [os.path.join(cropRoot, item) for item in
                    totalGribFileList[:int(0.8 * len(totalGribFileList))]]




    res = []

    for fileName in tqdm.tqdm(cropFileList[:], total=len(cropFileList),
                              desc="FileName", ncols=80, leave=False):
        with open(fileName, 'rb') as f:
            oneCropDataDict = pickle.load(f)
        # print(oneCropDataDict)
        for cropDataArray in tqdm.tqdm(oneCropDataDict.values(), total=len(oneCropDataDict),
                                       desc="One File Data", ncols=80, leave=False):
            # precipitationArray = cropDataArray.loc['Total precipitation', :, :]
            micapsValue = cropDataArray.attrs["micapValues"]
            # gribValue = np.mean(precipitationArray.values[8:10, 8:10]) * 1000
            # gapValue = gribValue - micapsValue
            res.append(micapsValue)

    for k,g in groupby(sorted(res),key=lambda x: x//1):
        print("{}: {}".format(k*1,len(list(g))))



if __name__ == "__main__":
    rangeStatistics()
