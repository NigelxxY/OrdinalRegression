import os
import tqdm
import torch
import pickle
import random
import datetime
import numpy as np
import torch.nn.functional as F
from torch.utils import data
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class gribDataset(data.Dataset):
    def __init__(self, args, transform, isTrain=True, model=None):
        self.cropDataFileRoot = args.cropRoot
        self.startDatetime = datetime.datetime.strptime(args.startDate, "%Y%m%d")
        self.endDatetime = datetime.datetime.strptime(args.endDate, "%Y%m%d")
        self.trainTestFileSplit(args.seed, isTrain, args.cross)
        self.isTrain = isTrain
        self.nClass = args.nClass
        self.initCenter = np.array([[50.0], [30.0], [10.0], [0.2]])
        self.loadCropData()
        self.transformer = transform
        self.oneHotClass = np.eye(self.nClass, self.nClass).astype(np.float32)
        self.rainOneHotClass = np.eye(2, 2).astype(np.float32)
        self.smallAxis = np.arange(0, 2.1, 0.25)
        self.largeAxis = np.arange(0, 2.1, 0.125)
        if isTrain:
            self.labelsKmeansTrain()
        else:
            self.cls = model
            self.labelsKmeansPredict()

    # load data for train or test
    def trainTestFileSplit(self, index, isTrain, cross=0):
        random.seed(index)
        totalGribFileList = os.listdir(self.cropDataFileRoot)
        random.shuffle(totalGribFileList)
        length = len(totalGribFileList)
        if isTrain:
            self.cropDataFileNamesList = [os.path.join(self.cropDataFileRoot, item) for item in
                                          (list(totalGribFileList[0:int(0.2 * cross * length)]) + list(
                                              totalGribFileList[int(0.2 * (cross + 1) * length):length]))]
        else:
            self.cropDataFileNamesList = [os.path.join(self.cropDataFileRoot, item) for item in
                                          totalGribFileList[int(0.2 * length * cross):int(0.2 * length * (cross + 1))]]

    def loadCropData(self):
        self.cropDataList = []
        for pickleFileName in tqdm.tqdm(self.cropDataFileNamesList, total=len(self.cropDataFileNamesList),
                                        desc="Load {} Data".format(self.isTrain), ncols=80, leave=False):
            cropDataDict = self.openPickleFile(pickleFileName)
            for cropDataArray in cropDataDict.values():
                # precipitationArray = cropDataArray.loc['Total precipitation', :, :]
                micapsValue = cropDataArray.attrs["micapValues"]

                rainFallClass = 0
                if micapsValue >= 0.1:
                    rainFallClass = 1
                # if micapsValue >= 36:
                #     continue

                # gridPredictedValue = np.mean(precipitationArray.values[8:10, 8:10]) * 1000
                # residualValue = abs(gridPredictedValue-micapsValue)

                cropData = (cropDataArray.values, micapsValue, rainFallClass)
                self.cropDataList.append(cropData)

    def openPickleFile(self, pickleFileName):
        if not os.path.exists(pickleFileName):
            raise NameError("No %s file exists." % (pickleFileName))

        with open(pickleFileName, 'rb') as f:
            oneCropData = pickle.load(f)

        return oneCropData

    def labelsKmeansTrain(self):
        # labels = [[item[1]] for item in self.cropDataList]
        labels = {i: [item[1]] for i, item in enumerate(self.cropDataList)}
        self.cls = KMeans(n_clusters=self.nClass, random_state=0)
        self.cls.fit(list(labels.values()))

        sortedIndex = np.argsort(self.cls.cluster_centers_.reshape(-1))[::-1]
        labelsMap = dict(zip(list(sortedIndex), list(range(self.nClass))))
        cachedlabelsClass = dict(zip(labels.keys(), self.cls.labels_))
        self.labelsClass = {i: labelsMap[item] for i, item in cachedlabelsClass.items()}

        cachedlabelsvalues = {i: [] for i in range(self.nClass)}
        for idx, labelClass in self.labelsClass.items():
            cachedlabelsvalues[labelClass].append(self.cropDataList[idx][1])

        self.means = [np.mean(cachedlabelsvalues[i]) for i in range(self.nClass)]
        # self.means.append(0)
        self.std = [np.std(cachedlabelsvalues[i]) for i in range(self.nClass)]
        # self.std.append(1)

    def labelGmmTrain(self):
        labels = {i: [item[1]] for i, item in enumerate(self.cropDataList) if item[1] >= 0.1}
        self.gmmModel = GaussianMixture(n_components=self.nClass, max_iter=500, random_state=0)
        self.gmmModel.fit(list(labels.values()))
        sortIndex = np.argsort(self.gmmModel.means_.reshape(-1))[::-1]
        trainProds = self.gmmModel.predict_proba(list(labels.values()))[:, sortIndex]
        trianLabels = np.argmax(trainProds, axis=1)
        self.labelsClass = dict(zip(labels.keys(), trianLabels))
        self.prodsDict = dict(zip(labels.keys(), trainProds))
        self.means = self.gmmModel.means_.reshape(-1)[sortIndex]
        self.means = np.append(self.means, 0)
        self.std = np.sqrt(self.gmmModel.covariances_.reshape(-1)[sortIndex])
        self.std = np.append(self.std, 1)

    def labelsKmeansPredict(self):
        labels = {i: [item[1]] for i, item in enumerate(self.cropDataList)}
        predictedClass = self.cls.predict(list(labels.values()))
        sortedIndex = np.argsort(self.cls.cluster_centers_.reshape(-1))[::-1]
        labelsMap = dict(zip(list(sortedIndex), list(range(self.nClass))))
        cachedlabelsClass = dict(zip(labels.keys(), predictedClass))
        self.labelsClass = {i: labelsMap[item] for i, item in cachedlabelsClass.items()}

    def labelGmmPredict(self):
        labels = {i: [item[1]] for i, item in enumerate(self.cropDataList) if item[1] >= 0.1}
        sortIndex = np.argsort(self.gmmModel.means_.reshape(-1))[::-1]
        predictProds = self.gmmModel.predict_proba(list(labels.values()))[:, sortIndex]
        predictLabels = np.argmax(predictProds, axis=1)
        self.labelsClass = dict(zip(labels.keys(), predictLabels))
        self.prodsDict = dict(zip(labels.keys(), predictProds))

    def centorCrop(self, inputs, cropSize):
        _, x, y = inputs.shape
        startX = x // 2 - cropSize // 2
        startY = y // 2 - cropSize // 2
        return inputs[:, startX:startX + cropSize, startY:startY + cropSize]

    # def resize(self, inputs, reSize=17):
    #     return F.interpolate(inputs,size=reSize,mode='bilinear')

    def __len__(self):
        return len(self.cropDataList)

    def __getitem__(self, idx):
        oneData = self.cropDataList[idx]
        features = torch.from_numpy(oneData[0].astype(np.float32))
        # features = self.resize(features,reSize=17)
        micapsValue = np.array([oneData[1]], dtype=np.float32)

        rainFallClass = np.array(oneData[2], dtype=np.int64)
        rainFallMask = self.rainOneHotClass[oneData[2]]

        regressionClassIndex = self.labelsClass[idx]
        regressionClass = np.array(regressionClassIndex, dtype=np.int64)
        regressionMask = self.oneHotClass[regressionClassIndex]

        if self.transformer:
            features = self.transformer(features)

        # gridPvalues = torch.from_numpy(oneData[3].astype(np.float32)).unsqueeze(0)
        # classIndex = self.labelsClass[idx] if idx in self.labelsClass.keys() else self.nClass
        # regressionMask = self.oneHotClass[classIndex]
        # logits = np.array(classIndex, dtype=np.int64)
        # regressionMask = np.append(self.prodsDict[idx], 0).astype(np.float32) if idx in self.prodsDict.keys() else \
        #     self.oneHotClass[self.nClass]
        # logits = np.array(self.labelsClass[idx], dtype=np.int64) if idx in self.labelsClass.keys() else np.array(
        #     self.nClass, dtype=np.int64)

        return features, micapsValue, rainFallClass, rainFallMask, regressionClass, regressionMask


class regressionGribDataset(data.Dataset):
    def __init__(self, args, transform):
        self.cropDataFileRoot = args.cropRoot
        self.startDatetime = datetime.datetime.strptime(args.startDate, "%Y%m%d")
        self.endDatetime = datetime.datetime.strptime(args.endDate, "%Y%m%d")
        self.trainTestFileSplit(args.seed)
        self.nClass = args.nClass
        self.initCenter = np.array([[50.0], [30.0], [10.0], [0.2]])
        self.loadCropData()
        self.transformer = transform
        self.oneHotClass = np.eye(self.nClass, self.nClass).astype(np.float32)
        self.rainOneHotClass = np.eye(2, 2).astype(np.float32)
        self.labelsKmeansTrain()

    # load data for train or test
    def trainTestFileSplit(self, index):
        random.seed(index)
        totalGribFileList = os.listdir(self.cropDataFileRoot)
        random.shuffle(totalGribFileList)
        self.cropDataFileNamesList = [os.path.join(self.cropDataFileRoot, item) for item in
                                      totalGribFileList[:int(0.8 * len(totalGribFileList))]]

    def loadCropData(self):
        self.cropDataList = []
        for pickleFileName in tqdm.tqdm(self.cropDataFileNamesList, total=len(self.cropDataFileNamesList),
                                        desc="Load {} Data".format("rainFall"), ncols=80, leave=False):
            cropDataDict = self.openPickleFile(pickleFileName)
            for cropDataArray in cropDataDict.values():
                # precipitationArray = cropDataArray.loc['Total precipitation', :, :]
                micapsValue = cropDataArray.attrs["micapValues"]

                rainFallClass = 1
                if micapsValue < 0.1:
                    continue
                # gridPredictedValue = np.mean(precipitationArray.values[8:10, 8:10]) * 1000
                # residualValue = abs(gridPredictedValue-micapsValue)
                # if residualValue > 15:
                #     continue
                # residualValue = gribValue-micapsValue

                cropData = (cropDataArray.values, micapsValue, rainFallClass)
                self.cropDataList.append(cropData)

    def openPickleFile(self, pickleFileName):
        if not os.path.exists(pickleFileName):
            raise NameError("No %s file exists." % (pickleFileName))

        with open(pickleFileName, 'rb') as f:
            oneCropData = pickle.load(f)

        return oneCropData

    def labelsKmeansTrain(self):
        # labels = [[item[1]] for item in self.cropDataList]
        labels = {i: [item[1]] for i, item in enumerate(self.cropDataList)}
        self.cls = KMeans(n_clusters=self.nClass, random_state=0)
        self.cls.fit(list(labels.values()))

        sortedIndex = np.argsort(self.cls.cluster_centers_.reshape(-1))[::-1]
        labelsMap = dict(zip(list(sortedIndex), list(range(self.nClass))))
        cachedlabelsClass = dict(zip(labels.keys(), self.cls.labels_))
        self.labelsClass = {i: labelsMap[item] for i, item in cachedlabelsClass.items()}

        cachedlabelsvalues = {i: [] for i in range(self.nClass)}
        for idx, labelClass in self.labelsClass.items():
            cachedlabelsvalues[labelClass].append(self.cropDataList[idx][1])

        self.means = [np.mean(cachedlabelsvalues[i]) for i in range(self.nClass)]
        print(self.means)
        self.std = [np.std(cachedlabelsvalues[i]) for i in range(self.nClass)]
        print(self.std)

    def labelGmmTrain(self):
        labels = {i: [item[1]] for i, item in enumerate(self.cropDataList) if item[1] >= 0.1}
        self.gmmModel = GaussianMixture(n_components=self.nClass, max_iter=500, random_state=0)
        self.gmmModel.fit(list(labels.values()))
        sortIndex = np.argsort(self.gmmModel.means_.reshape(-1))[::-1]
        trainProds = self.gmmModel.predict_proba(list(labels.values()))[:, sortIndex]
        trianLabels = np.argmax(trainProds, axis=1)
        self.labelsClass = dict(zip(labels.keys(), trianLabels))
        self.prodsDict = dict(zip(labels.keys(), trainProds))
        self.means = self.gmmModel.means_.reshape(-1)[sortIndex]
        self.means = np.append(self.means, 0)
        self.std = np.sqrt(self.gmmModel.covariances_.reshape(-1)[sortIndex])
        self.std = np.append(self.std, 1)

    def __len__(self):
        return len(self.cropDataList)

    def __getitem__(self, idx):
        oneData = self.cropDataList[idx]
        features = torch.from_numpy(oneData[0].astype(np.float32))
        # features = self.resize(features,reSize=17)
        micapsValue = np.array([oneData[1]], dtype=np.float32)

        rainFallClass = np.array(oneData[2], dtype=np.int64)
        rainFallMask = self.rainOneHotClass[oneData[2]]

        regressionClassIndex = self.labelsClass[idx]
        regressionClass = np.array(regressionClassIndex, dtype=np.int64)
        regressionMask = self.oneHotClass[regressionClassIndex]

        if self.transformer:
            features = self.transformer(features)

        # gridPvalues = torch.from_numpy(oneData[3].astype(np.float32)).unsqueeze(0)
        # classIndex = self.labelsClass[idx] if idx in self.labelsClass.keys() else self.nClass
        # regressionMask = self.oneHotClass[classIndex]
        # logits = np.array(classIndex, dtype=np.int64)
        # regressionMask = np.append(self.prodsDict[idx], 0).astype(np.float32) if idx in self.prodsDict.keys() else \
        #     self.oneHotClass[self.nClass]
        # logits = np.array(self.labelsClass[idx], dtype=np.int64) if idx in self.labelsClass.keys() else np.array(
        #     self.nClass, dtype=np.int64)

        return features, micapsValue, rainFallClass, rainFallMask, regressionClass, regressionMask


class ordinalGribDataset(data.Dataset):
    def __init__(self, args, transform, range):
        self.cropDataFileRoot = args.cropRoot
        self.startDatetime = datetime.datetime.strptime(args.startDate, "%Y%m%d")
        self.endDatetime = datetime.datetime.strptime(args.endDate, "%Y%m%d")
        self.trainTestFileSplit(args.seed, args.cross)
        self.nClass = args.nClass
        self.space = 0.5
        self.range = range
        self.loadCropData()
        self.transformer = transform

    # load data for train or test
    def trainTestFileSplit(self, index, cross=0):
        random.seed(index)
        totalGribFileList = os.listdir(self.cropDataFileRoot)
        random.shuffle(totalGribFileList)
        length = len(totalGribFileList)
        self.cropDataFileNamesList = [os.path.join(self.cropDataFileRoot, item) for item in
                                      (list(totalGribFileList[0:int(0.2 * cross * length)]) + list(
                                          totalGribFileList[int(0.2 * (cross + 1) * length):length]))]

    def loadCropData(self):
        self.cropDataList = []
        for pickleFileName in tqdm.tqdm(self.cropDataFileNamesList, total=len(self.cropDataFileNamesList),
                                        desc="Load {} Data".format("rainFall"), ncols=80, leave=False):
            cropDataDict = self.openPickleFile(pickleFileName)
            for cropDataArray in cropDataDict.values():
                # precipitationArray = cropDataArray.loc['Total precipitation', :, :]
                micapsValue = cropDataArray.attrs["micapValues"]

                if micapsValue < 0.1:
                    # micapsValue = 0
                    continue

                ordinalLabels = 1.0 * (self.range < micapsValue)

                cropData = (cropDataArray.values, micapsValue, ordinalLabels)
                self.cropDataList.append(cropData)

    def openPickleFile(self, pickleFileName):
        if not os.path.exists(pickleFileName):
            raise NameError("No %s file exists." % (pickleFileName))

        with open(pickleFileName, 'rb') as f:
            oneCropData = pickle.load(f)

        return oneCropData

    def __len__(self):
        return len(self.cropDataList)

    def __getitem__(self, idx):
        oneData = self.cropDataList[idx]
        features = torch.from_numpy(oneData[0].astype(np.float32))
        # features = self.resize(features,reSize=17)
        micapsValue = np.array([oneData[1]], dtype=np.float32)
        ordinalLabels = oneData[2].astype(np.float32)

        if self.transformer:
            features = self.transformer(features)

        return features, micapsValue, ordinalLabels
