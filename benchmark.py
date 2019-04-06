import os
import pickle
import tqdm
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from collections import Counter


def calMean(gapList):
    errorValues = np.mean(gapList)
    return errorValues
    # print("Error values: ",errorValues)


def ts(predicted, labels, threshold):
    tp = np.sum((predicted >= threshold) * (labels >= threshold))
    tn = np.sum((predicted < threshold) * (labels < threshold))
    fp = np.sum((predicted < threshold) * (labels >= threshold))
    fn = np.sum((predicted >= threshold) * (labels < threshold))

    ts = round(tp / (tp + fp + fn), 5)
    return ts


def calPmae(predicted, labels):
    gaps = np.abs(predicted - labels)
    pCount = np.sum(labels > 0.05)
    pError = np.sum((labels > 0.05) * gaps)
    mpae = round(pError / pCount, 5)
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

    lr = LinearRegression()
    svr = SVR(gamma=0.001, C=1.0, epsilon=0.2)

    featuresList = []
    labels = []

    for fileName in tqdm.tqdm(cropFileList[:], total=len(cropFileList),
                              desc="FileName", ncols=80, leave=False):
        with open(fileName, 'rb') as f:
            oneCropDataDict = pickle.load(f)
        # print(oneCropDataDict)
        for cropDataArray in tqdm.tqdm(oneCropDataDict.values(), total=len(oneCropDataDict),
                                       desc="One File Data", ncols=80, leave=False):
            features = np.mean(cropDataArray.values[:, 8:10, 8:10], axis=(1, 2))
            featuresList.append(features)

            micapsValue = cropDataArray.attrs["micapValues"]
            labels.append(micapsValue)

    featuresList = np.array(featuresList)
    labels = np.array(labels)
    # print(labels)
    # print(labels.shape)
    lr.fit(featuresList, labels)
    sampleIndex = np.random.choice(len(featuresList), size=10000, replace=False)
    featuresList = featuresList[sampleIndex, :]
    labels = labels[sampleIndex]
    print(featuresList)
    print(labels)
    svr.fit(featuresList, labels)

    testFeaturesList = []
    testLabelsList = []
    testGridPrecipitationList = []

    for fileName in tqdm.tqdm(testCropFileList[:], total=len(testCropFileList),
                              desc="FileName", ncols=80, leave=False):
        with open(fileName, 'rb') as f:
            oneCropDataDict = pickle.load(f)
        # print(oneCropDataDict)
        for cropDataArray in tqdm.tqdm(oneCropDataDict.values(), total=len(oneCropDataDict),
                                       desc="One File Data", ncols=80, leave=False):
            features = np.mean(cropDataArray.values[:, 8:10, 8:10], axis=(1, 2))
            testFeaturesList.append(features)

            micapsValue = cropDataArray.attrs["micapValues"]
            testLabelsList.append(micapsValue)

            gridValue = np.mean(cropDataArray.loc['Total precipitation', :, :].values[8:10, 8:10]) * 1000
            testGridPrecipitationList.append(gridValue)

    testFeaturesList = np.array(testFeaturesList)
    testLabelsList = np.array(testLabelsList)
    testGridPrecipitationList = np.array(testGridPrecipitationList)

    print("Bilinear Interplotaion")
    print("MAE : ", np.mean(np.abs(testGridPrecipitationList - testLabelsList)))
    print("MPAE : ", calPmae(predicted=testGridPrecipitationList, labels=testLabelsList))
    print("Ts0.1 : ", ts(predicted=testGridPrecipitationList, labels=testLabelsList, threshold=0.1))
    print("Ts1 : ", ts(predicted=testGridPrecipitationList, labels=testLabelsList, threshold=1))
    print("Ts10 : ", ts(predicted=testGridPrecipitationList, labels=testLabelsList, threshold=10))

    print("\n")
    #
    print("Linear Regression")
    testLrList = lr.predict(testFeaturesList)
    print("MAE : ", np.mean(np.abs(testLrList - testLabelsList)))
    print("MPAE : ", calPmae(predicted=testLrList, labels=testLabelsList))
    print("Ts0.1 : ", ts(predicted=testLrList, labels=testLabelsList, threshold=0.1))
    print("Ts1 : ", ts(predicted=testLrList, labels=testLabelsList, threshold=1))
    print("Ts10 : ", ts(predicted=testLrList, labels=testLabelsList, threshold=10))

    print("\n")

    print("Svr")
    testSvrList = svr.predict(testFeaturesList)
    print("MAE : ", np.mean(np.abs(testSvrList - testLabelsList)))
    print("MPAE : ", calPmae(predicted=testSvrList, labels=testLabelsList))
    print("Ts0.1 : ", ts(predicted=testSvrList, labels=testLabelsList, threshold=0.1))
    print("Ts1 : ", ts(predicted=testSvrList, labels=testLabelsList, threshold=1))
    print("Ts10 : ", ts(predicted=testSvrList, labels=testLabelsList, threshold=10))

    # precipitationArray = cropDataArray.loc['Total precipitation', :, :]
    # micapsValue = cropDataArray.attrs["micapValues"]
    # gribValue = np.mean(precipitationArray.values[8:10, 8:10]) * 1000
    # gapValue = gribValue - micapsValue
    # if micapsValue > 0.01:
    #     res.append([micapsValue])
    # cls = KMeans(n_clusters=4, random_state=0).fit(res)
    # counter = Counter(cls.labels_)
    # centers = cls.cluster_centers_
    # print(centers)
    # print(counter)
    #     gmm = GaussianMixture(n_components=4,max_iter=500)
    #     trainLabels = gmm.fit_predict(res)
    # print(trainLabels)
    # print(gmm.means_)
    # print(gmm.covariances_)
    #     sortIndex = np.argsort(gmm.means_.reshape(-1))[::-1]
    # print(sortIndex)
    # print(gmm.means_.reshape(-1)[sortIndex])
    # print(np.sqrt(gmm.covariances_.reshape(-1))[sortIndex])

    # gripRes = []
    # testRes = []

    correct = 0
    #

    # labels = cls.predict(gripRes)
    # prods = cls.predict(testRes)

    # print(testRes[:10])
    # print(labels)
    # print(prods)
    # print(prods[:,sortIndex])
    # print(np.argmax(prods[:,sortIndex],axis=1))
    # predicted = np.argmax(labels,axis=1)
    # logits = np.argmax(prods,axis=1)
    # correct += np.sum(labels == prods)
    # print(round(correct / len(oneCropDataDict), 5))


#
# labelOne = cls.predict(testRes)
# res = {i:[] for i in range(4)}
# for i,label in enumerate(labelOne):
#     bias = abs(centers[label] - testRes[i][0])
#     res[label].append(bias)

# for key,values in res.items():
#     print("{}:{} error is {}".format(key,centers[key],np.mean(values)))
# labelOne = np.array(cls.predict(gripRes))
# labelTwo = np.array(cls.predict(testRes))
#
# correct = np.sum((labelOne == labelTwo))
# total = len(labelOne)
# accuracy = correct/total
#
# print(correct)
# print(total)
# print(accuracy)






# cls = KMeans(n_clusters=4, random_state=0).fit(res)
# counter = Counter(cls.labels_)
# print(cls.cluster_centers_)
# print(counter)
#
# cls = KMeans(n_clusters=5, random_state=0).fit(res)
# counter = Counter(cls.labels_)
# print(cls.cluster_centers_)
# print(counter)
#
# cls = KMeans(n_clusters=6, random_state=0).fit(res)
# counter = Counter(cls.labels_)
# print(cls.cluster_centers_)
# print(counter)


if __name__ == "__main__":
    benchmarkGenerator()
