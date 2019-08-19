import os
import math
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

smallAxis = np.arange(0, 2.1, 0.25)
largeAxis = np.arange(0, 2.1, 0.125)


class workerDataset(data.Dataset):
    def __init__(self, args, flag, transformer):
        self.dataRoot = args.data_path
        self.transformer = transformer
        self.startTimes = datetime.datetime.strptime(self.dataRoot.split('/')[-1][-10:], "%Y%m%d%H")
        self.preTimes = self.startTimes + datetime.timedelta(
            hours=6 * flag)
        self.nowTimes = self.startTimes + datetime.timedelta(
            hours=6 * (flag + 1))
        print(self.preTimes)
        print(self.nowTimes)

        self.latRange = np.arange(23, 38.51, 0.05)
        self.lonRange = np.arange(113, 122.51, 0.05)

        self.data = []
        self.loc = []

    def __len__(self):
        len(self.data)

    def __getitem__(self, idx):
        features = self.data[idx]
        locs = self.loc[idx]
        features = torch.from_numpy(features.astype(np.float32))
        features = self.transformer(features)

        return features, locs

    def dataReader(self):
        startTimesStr = self.startTimes.strftime("%m%d%H")
        preTimesStr = self.preTimes.strftime("%m%d%H")
        nowTimesStr = self.nowTimes.strftime("%m%d%H")

        preFileName = os.path.join(self.dataRoot, "C1D" + startTimesStr + "00" + preTimesStr + "001")
        nowFileName = os.path.join(self.dataRoot, "C1D" + startTimesStr + "00" + nowTimesStr + "001")

        print(preFileName)
        print(nowFileName)

        preGribData = pygrib.open(preFileName)
        prePrecipitationData = [preGribData.select(name=keyName)[0] for keyName in precipitationKeys]

        gribData = pygrib.open(nowFileName)
        normalData = [gribData.select(name=keyName, level=level)[0] for keyName in normalKeys for level in levelsList]
        normalData += [gribData.select(name=keyName)[0] for keyName in singleLevelKeys]
        precipitationData = [gribData.select(name=keyName)[0] for keyName in precipitationKeys]

        for lat in self.latRange:
            for lon in self.lonRange:
                latLowerBound = math.floor(lat * 4) / 4 - 1
                lonLowerBound = math.floor(lon * 4) / 4 - 1
                latUpperBound = latLowerBound + 2
                lonUpperBound = lonLowerBound + 2

                featureValuesList = []

                for grib in normalData:
                    featureValues = \
                        grib.data(lat1=latLowerBound, lat2=latUpperBound, lon1=lonLowerBound, lon2=lonUpperBound)[0]
                    if featureValues.shape[0] < len(largeAxis):
                        featureValues = interpolate.interp2d(smallAxis, smallAxis, featureValues, kind='linear')(
                            largeAxis,
                            largeAxis)
                    featureValuesList.append(featureValues)

                latLowerBound = math.floor(lat * 8) / 8 - 1
                lonLowerBound = math.floor(lon * 8) / 8 - 1
                latUpperBound = latLowerBound + 2.01
                lonUpperBound = lonLowerBound + 2.01

                precipitationValuesList = [
                    grib.data(lat1=latLowerBound, lat2=latUpperBound, lon1=lonLowerBound, lon2=lonUpperBound)[0] for
                    grib in
                    precipitationData]
                prePrecipitationValuesList = [
                    grib.data(lat1=latLowerBound, lat2=latUpperBound, lon1=lonLowerBound, lon2=lonUpperBound)[0] for
                    grib in
                    prePrecipitationData]
                precipitationValuesList = list(
                    map(lambda x: x[1] - x[0], zip(prePrecipitationValuesList, precipitationValuesList)))

                featureValuesList += precipitationValuesList

                featureValues = np.stack(featureValuesList)

                self.data.append(featureValues)
                self.loc.append((lat, lon))


class Worker():
    def __init__(self, args, nClass):
        self.args = args
        self.nClass = args.nClass

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.autoencoder = AutoencoderBN().to(self.device)
        self.regressor = OrdinalRegressionModel(self.nClass).to(self.device)
        self.classification = rainFallClassification().to(self.device)

    def initModel(self):
        modelsParam = torch.load(self.args.modelFile, map_location=self.device)

        self.autoencoder.load_state_dict(modelsParam["modelParam"])
        self.autoencoder.eval()
        print("================ Load Autoencoder Done ! ================")

        self.regressor.load_state_dict(modelsParam["regressionParam"])
        self.regressor.eval()
        print("================ Load Regressor Done ! ================")

        self.classification.load_state_dict(modelsParam["rainFallParam"])
        self.classification.eval()
        print("================ Load RainFallClassification Done ! ================")

    def generateOneHot(self, softmax):
        maxIdxs = torch.argmax(softmax, dim=1, keepdim=True).cpu().long()
        oneHotMask = torch.zeros(softmax.shape, dtype=torch.float32)
        oneHotMask = oneHotMask.scatter_(1, maxIdxs, 1.0)
        oneHotMask = oneHotMask.unsqueeze(-2)
        return oneHotMask

    def validate(self, flag, dataLoader):
        for features, locs in dataLoader:
            features = features.to(self.device)
            locs = locs.to(self.device)

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

                regressionValues = regressionValues.item()
                print(locs)
                print(regressionValues)


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
    for flag in range(1, 11):
        oneDataset = workerDataset(args, flag, transformer)
        oneDataloader = data.DataLoader(oneDataset, batch_size=256)
        worker.validate(flag, oneDataloader)


if __name__ == "__main__":
    import argparse

    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="/public/data/ECTHIN_C1D", type=str)
    parser.add_argument("--worker_path", default="/public/home/cmacontest/fd_correction/workerDir", type=str)
    parser.add_argument("--model_path",default="/public/home/cmacontest/fd_correction/modelDir",type=str)
    parser.add_argument("--nClass", default=70, type=int)

    args = parser.parse_args()

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

    args.modelFile = os.path.join(args.worker_path,"worker_checkpoints.pth")

    jobFunc(args)
