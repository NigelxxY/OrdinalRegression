import os
import tqdm
import math
import torch
import random
import numpy as np
from torch.utils import data

root_path = "/mnt/pami14/DATASET/METEOROLOGY/ECforecast/GRIB"
micaps_path = "/mnt/pami14/DATASET/METEOROLOGY/GT_micaps"
features_save_path = "/mnt/pami14/xyxu/dataset/ecCropData/features"
labels_save_path = "/mnt/pami14/xyxu/dataset/ecCropData/labels"

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

locBound = (23.0, 38.51, 110, 126.51)
smallLatAxis = np.arange(0, locBound[1] - locBound[0] + 2, 0.25)
smallLonAxis = np.arange(0, locBound[3] - locBound[2] + 2, 0.25)
largeLatAxis = np.arange(0, locBound[1] - locBound[0] + 2, 0.125)
largeLonAxis = np.arange(0, locBound[3] - locBound[2] + 2, 0.125)


class Dataset(data.Dataset):
    def __init__(self, args, transform, range, isTrain=True, isRain=True):
        self.feature_save_path = args.featurePath
        self.label_save_path = args.labelPath
        self.isRain = isRain
        self.transformer = transform
        self.range = range
        self.trainTestFileSplit(index=0, isTrain=isTrain, cross=args.cross)
        self.loadFeatureLabels()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        key = self.labelsKey[idx]
        label = self.labels[idx]
        index, lat, lon, micapsValue = label
        latLowerBound = math.floor(lat * 8) / 8 - 1
        lonLowerBound = math.floor(lon * 8) / 8 - 1
        latLowerIndex = int(latLowerBound - locBound[0] - 1) * 8
        lonLowerIndex = int(lonLowerBound - locBound[2] - 1) * 8
        features = self.features[key][:, latLowerIndex:latLowerIndex + 17, lonLowerIndex:lonLowerIndex + 17]
        # print(features.shape)
        features = torch.from_numpy(features.astype(np.float32))
        if self.transformer:
            features = self.transformer(features)

        micapsValue = np.array([micapsValue],dtype=np.float32)
        rainLabel = np.array(1 * (micapsValue > 1e-2),dtype=np.int64)
        ordinalLabel = np.array(1.0 * (self.range < micapsValue),dtype=np.float32)
        return features, micapsValue, rainLabel, ordinalLabel

    def trainTestFileSplit(self, index, isTrain, cross=0):
        random.seed(index)
        featuresFileList = os.listdir(self.feature_save_path)
        labelsFileList = os.listdir(self.label_save_path)
        commonFiles = list(set(featuresFileList) & set(labelsFileList))
        random.shuffle(commonFiles)
        length = len(commonFiles)
        if isTrain:
            self.commonFileList = list(commonFiles[0:int(0.2 * cross * length)]) + list(
                commonFiles[int(0.2 * (cross + 1) * length):length])
        else:
            self.commonFileList = commonFiles[int(0.2 * length * cross):int(0.2 * length * (cross + 1))]

    def loadFeatureLabels(self):
        self.features = {}
        self.labelsKey = []
        self.labels = []

        for featureFile in tqdm.tqdm(self.commonFileList[:], total=len(self.commonFileList),
                                     desc='Load Features', ncols=80, leave=True):
            featureFullName = os.path.join(self.feature_save_path, featureFile)
            saveKey = featureFile.split('.')[0]
            labelFullName = os.path.join(self.label_save_path, featureFile)

            feature = np.load(featureFullName)
            label = np.load(labelFullName)
            if label.shape[0] < 1:
                continue

            if self.isRain:
                mask = label[:, 3] > 1e-2
                label = label[mask]

            labelSize = label.shape[0]
            if labelSize < 1:
                continue
            self.features[saveKey] = feature
            self.labelsKey.extend([saveKey] * labelSize)
            self.labels.append(label)

        self.labels = np.concatenate(self.labels, axis=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--featurePath", default=features_save_path, type=str)
    parser.add_argument("--labelPath", default=labels_save_path, type=str)

    args = parser.parse_args()
    print(args)

    dataset = Dataset(args)
    # print(len(dataset))
    # for _ in range(20):
    #     idx = random.randint(0,len(dataset))
    #     print(dataset[idx])
