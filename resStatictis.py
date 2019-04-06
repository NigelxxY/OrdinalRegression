import os
import torch
import pickle
import random
import datetime
import numpy as np
from torch.utils import data
from sklearn.cluster import KMeans


def openPickleFile(pickleFileName):
    if not os.path.exists(pickleFileName):
        raise NameError("No %s file exists." % (pickleFileName))

    with open(pickleFileName, 'rb') as f:
        oneCropData = pickle.load(f)

    return oneCropData

def dataloader(args):
    cropDataFileRoot = args.cropRoot
    random.seed(args.seed)
    totalGribFileList = os.listdir(cropDataFileRoot)
    random.shuffle(totalGribFileList)
    cropDataFileNamesList = [os.path.join(cropDataFileRoot, item) for item in
                             totalGribFileList[:int(0.8 * len(totalGribFileList))]]
    cropDataList = []
    for pickleFileName in cropDataFileNamesList:
        cropDataDict = openPickleFile(pickleFileName)
        for cropDataArray in cropDataDict.values():

            ecData = cropDataArray.sel(Key="Total precipitation").values[8:9,8:9]
            micapsValues = cropDataArray.attrs["micapValues"]

            cropDataList.append(cropData)
