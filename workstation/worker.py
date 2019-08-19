import os
import math
import time
import pygrib
import torch
import datetime
import numpy as np
import torch.nn as nn
from torch.utils import data
from torch.nn import functional as F
from torchvision import transforms
from scipy import interpolate
from model import AutoencoderBN, OrdinalRegressionModel, rainFallClassification

normalKeys = ["Relative humidity",
              "Vertical velocity",
              "Temperature",
              "V component of wind",
              "U component of wind",
              "Potential vorticity",
              "Divergence",
              "Geopotential Height"
              ]

singleLevelKeys = ["Convective available potential energy",
                   "Total column water",
                   "Total cloud cover"]

precipitationKeys = ["Large-scale precipitation",
                     "Total precipitation"
                     ]

levelsList = [500, 700, 850, 925]

locBound = (23.0, 38.51, 113, 122.51)
smallLatAxis = np.arange(0, locBound[1] - locBound[0] + 2, 0.25)
smallLonAxis = np.arange(0, locBound[3] - locBound[2] + 2, 0.25)
largeLatAxis = np.arange(0, locBound[1] - locBound[0] + 2, 0.125)
largeLonAxis = np.arange(0, locBound[3] - locBound[2] + 2, 0.125)


class workerDataset(data.Dataset):
    def __init__(self, args, flag, transformer):
        self.dataRoot = args.data_path
        self.transformer = transformer
        self.startTimes = datetime.datetime.strptime(self.dataRoot.split('/')[-1][-10:], "%Y%m%d%H")
        self.preTimes = self.startTimes + datetime.timedelta(
            hours=6 * flag)
        self.nowTimes = self.startTimes + datetime.timedelta(
            hours=6 * (flag + 1))
        self.slidingWindows = 17
        print(self.preTimes)
        print(self.nowTimes)

        self.locBound = locBound  # latLowerBound,latUpperBound,lonLowerBound,lonUpperBound

        self.latRange = np.arange(self.locBound[0], self.locBound[1], 0.125)[::-1]
        self.lonRange = np.arange(self.locBound[2], self.locBound[3], 0.125)

        self.data = []
        self.loc = []

        self.largeDataReader()
        self.dataSliding()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx]
        locs = self.loc[idx]
        features = torch.from_numpy(features.astype(np.float32))
        features = self.transformer(features)

        return features, locs

    def largeDataReader(self):
        startTimesStr = self.startTimes.strftime("%m%d%H")
        preTimesStr = self.preTimes.strftime("%m%d%H")
        nowTimesStr = self.nowTimes.strftime("%m%d%H")

        preFileName = os.path.join(self.dataRoot, "C1D" + startTimesStr + "00" + preTimesStr + "001")
        nowFileName = os.path.join(self.dataRoot, "C1D" + startTimesStr + "00" + nowTimesStr + "001")

        print(preFileName)
        print(nowFileName)

        beforeOpentime = time.time()

        preGribData = pygrib.open(preFileName)
        gribData = pygrib.open(nowFileName)

        prePrecipitationData = [preGribData.select(name=keyName)[0] for keyName in precipitationKeys]
        normalData = [gribData.select(name=keyName, level=level)[0] for keyName in normalKeys for level in levelsList]
        normalData += [gribData.select(name=keyName)[0] for keyName in singleLevelKeys]
        precipitationData = [gribData.select(name=keyName)[0] for keyName in precipitationKeys]

        featureValuesList = []

        for grib in normalData:
            featureValues = \
                grib.data(lat1=self.locBound[0] - 1, lat2=self.locBound[1] + 1, lon1=self.locBound[2] - 1,
                          lon2=self.locBound[3] + 1)[0]
            if featureValues.shape[0] < len(largeLatAxis):
                featureValues = interpolate.interp2d(smallLonAxis, smallLatAxis, featureValues, kind='linear')(
                    largeLonAxis,
                    largeLatAxis)

            featureValuesList.append(featureValues)

        precipitationValuesList = [
            grib.data(lat1=self.locBound[0] - 1, lat2=self.locBound[1] + 1, lon1=self.locBound[2] - 1,
                      lon2=self.locBound[3] + 1)[0] for grib in precipitationData]
        prePrecipitationValuesList = [
            grib.data(lat1=self.locBound[0] - 1, lat2=self.locBound[1] + 1, lon1=self.locBound[2] - 1,
                      lon2=self.locBound[3] + 1)[0] for grib in prePrecipitationData]
        precipitationValuesList = list(
            map(lambda x: x[1] - x[0], zip(prePrecipitationValuesList, precipitationValuesList)))

        featureValuesList += precipitationValuesList

        self.totalFeatures = np.stack(featureValuesList)

        print("Data Loadering Time Cost is ", time.time() - beforeOpentime)

    def dataSliding(self):

        for i, lat in enumerate(self.latRange):
            for j, lon in enumerate(self.lonRange):
                slidingFeatures = self.totalFeatures[:, i:i + self.slidingWindows, j:j + self.slidingWindows]
                self.data.append(slidingFeatures)
                self.loc.append((lat, lon))

        print("============ Feature Loading Done ! ============")


class Worker():
    def __init__(self, args):
        self.args = args
        self.nClass = args.nClass

        self.device = torch.device('cpu')
        self.autoencoder = AutoencoderBN().to(self.device)
        self.regressor = OrdinalRegressionModel(self.nClass).to(self.device)
        self.classification = rainFallClassification().to(self.device)

    def initModel(self):
        modelsParam = torch.load(self.args.modelFile, map_location=self.device)

        self.autoencoder.load_state_dict(modelsParam["modelParam"])
        self.autoencoder.eval()
        print("================= Load Autoencoder Done ===========================")

        self.regressor.load_state_dict(modelsParam["regressionParam"])
        self.regressor.eval()
        print("================= Load Regressor Done =============================")

        self.classification.load_state_dict(modelsParam["rainFallParam"])
        self.classification.eval()
        print("================= Load RainFallClassification Done ================")

    def generateOneHot(self, softmax):
        maxIdxs = torch.argmax(softmax, dim=1, keepdim=True).cpu().long()
        oneHotMask = torch.zeros(softmax.shape, dtype=torch.float32)
        oneHotMask = oneHotMask.scatter_(1, maxIdxs, 1.0)
        oneHotMask = oneHotMask.unsqueeze(-2)
        return oneHotMask

    def validate(self, savePath, dataLoader):
        correctionValuesList = []
        for features, locs in dataLoader:
            features = features.to(self.device)

            with torch.no_grad():
                encoder, _ = self.autoencoder(features)
                predictValues = self.regressor(encoder)
                rainPreds = self.classification(features)

                rainPredsSoftMax = F.softmax(rainPreds, dim=1)
                rainOnehot = self.generateOneHot(rainPredsSoftMax)

                regressionValues = 0.5 * (torch.sum((predictValues > 0.5).float(), dim=1).view(-1, 1))
                zeros = torch.zeros(regressionValues.size()).to(self.device)

                regressionValues = torch.matmul(rainOnehot,
                                                torch.cat([zeros, regressionValues], dim=1).unsqueeze(-1)).squeeze(-1)

                correctionValues = regressionValues.cpu().numpy().reshape(5, -1)
                correctionValuesList.append(correctionValues)

        res = np.concatenate(correctionValuesList)[::-1, ].reshape(-1)
        res.tofile(savePath)
        print("Save File at ", savePath)


def jobFunc(args):
    meanFilePath = os.path.join(args.worker_path, "mean.npy")
    stdFilePath = os.path.join(args.worker_path, "std.npy")
    featuresMean = np.load(meanFilePath)
    featuresStd = np.load(stdFilePath)
    transformer = transforms.Compose([
        transforms.Normalize(mean=featuresMean, std=featuresStd)
    ])

    worker = Worker(args)
    worker.initModel()
    saveFileDirName = os.path.join(args.save_path, args.data_path.split('/')[-1][-10:])
    if os.path.exists(saveFileDirName):
        print("Save Dir {} has existed. Check Again Please")
    else:
        os.mkdir(saveFileDirName)
        for flag in range(1, 11):
            oneDataset = workerDataset(args, flag, transformer)
            batchSize = len(oneDataset.lonRange) * 5
            saveFileName = os.path.join(saveFileDirName, oneDataset.preTimes.strftime("%m%d%H"))
            oneDataloader = data.DataLoader(oneDataset, batch_size=batchSize)
            worker.validate(saveFileName, oneDataloader)


if __name__ == "__main__":
    import argparse

    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="/public/data/ECTHIN_C1D", type=str)
    parser.add_argument("--worker_path", default="/public/home/cmacontest/fd_correction/workerDir", type=str)
    parser.add_argument("--model_path", default="/public/home/cmacontest/fd_correction/modelDir", type=str)
    parser.add_argument("--save_path", default="/public/home/cmacontest/fd_correction/resultDir", type=str)
    parser.add_argument("--nClass", default=70, type=int)

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    fileNameList = os.listdir(args.data_path)
    if len(fileNameList) > 1:
        now = datetime.datetime.now()
        minTimeDelta = datetime.timedelta(days=999)
        closeFileName = ""
        for fileName in fileNameList:
            fileDate = datetime.datetime.strptime(fileName[-10:], "%Y%m%d%H")
            if now - fileDate < minTimeDelta:
                minTimeDelta = now - fileDate
                closeFileName = fileName
        args.data_path = os.path.join(args.data_path, closeFileName)
    elif len(fileNameList) < 1:
        raise FileNotFoundError("EC DATA MISSING.")
    else:
        args.data_path = os.path.join(args.data_path, fileNameList[0])

    args.modelFile = os.path.join(args.worker_path, "worker_checkpoints.pth")

    jobFunc(args)
