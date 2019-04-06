import os
import sys
import tqdm
import math
import pygrib
import pickle
import datetime
import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate
from dateutil.relativedelta import relativedelta

root_path = "/mnt/pami/DATASET/METEOROLOGY/ECforecast/GRIB"
micaps_path = "/mnt/pami/DATASET/METEOROLOGY/GT_micaps"
save_path = "/mnt/pami/xyxu/dataset/cropGrib"

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


def testFunction(fileNameList, preFileNameDict):
    for fileName in fileNameList:
        if fileName in preFileNameDict.keys():
            print(fileName)
            print(preFileNameDict[fileName])
        else:
            print(fileName)
            print("XXXXXX")


def dataPreLoader(yearMonth):
    """
    Save Grib Data as csv File
    csv file name '%m%d%H.csv'
    columns: Feature Name_Level
    index: micaps index
    data: numpy array of feature
    :return: csv file name list
    """

    # Generate datetime tips for 12 hour gap
    fileNameList, preFileNameDict = fileNameListGenerator(yearMonth)
    # testFunction(fileNameList,preFileNameDict)
    for fileName in tqdm.tqdm(
            fileNameList, total=len(fileNameList),
            desc='Load Data', ncols=80,
            leave=False):

        # beforeOpenTime = time.time()
        gribData = pygrib.open(fileName)
        # afterOpenTime = time.time()

        # print("Open File Time Cost is {}".format(afterOpenTime-beforeOpenTime))

        # Load Grib Data with multi Level
        # beforeTime = time.time()
        normalData = [gribData.select(name=keyName, level=level)[0] for keyName in normalKeys for level in levelsList]
        # afterTime = time.time()
        # print("Multi Time Cost is {}".format(afterTime-beforeTime))


        # Load Grib Data with level 0
        normalData += [gribData.select(name=keyName)[0] for keyName in singleLevelKeys]

        # Load Precipitation Data
        precipitationData = [gribData.select(name=keyName)[0] for keyName in precipitationKeys]
        prePrecipitationData = None
        if fileName in preFileNameDict.keys():
            preGribData = pygrib.open(preFileNameDict[fileName])
            prePrecipitationData = [preGribData.select(name=keyName)[0] for keyName in precipitationKeys]

        # Load micaps point Data
        micapsFileName = searchMicapsFileByGribFileName(fileName)
        labelsList = getOneDateMicapsDataList(micapsFileName)

        # Crop Data
        # beforeTime = time.time()
        cropData = cropGribByMicapsData(labelsList, normalData, precipitationData, prePrecipitationData)
        pklName = os.path.join(save_path,fileName.split("/")[-3]+fileName.split("/")[-1][-9:-3]+".pkl")

        with open(pklName,"wb") as f:
            pickle.dump(cropData,f,protocol=pickle.HIGHEST_PROTOCOL)

def fileNameListGenerator(start_date):
    preFileNameDict = {}
    fileNameList = []
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y%m%d")
    end_date = start_date + relativedelta(months=1)
    date_list = [start_date + datetime.timedelta(hours=12 * x) for x in
                 range(0, 2 * (end_date - start_date).days)]

    for oneDate in date_list:
        yearStr = oneDate.strftime("%Y")
        mdhStr = oneDate.strftime("%m%d%H")
        file_dir_name = os.path.join(root_path, yearStr, mdhStr)
        if not os.path.exists(file_dir_name):
            continue

        preDeadLineDate = oneDate + datetime.timedelta(hours=6)
        postDeadLineDate = oneDate + datetime.timedelta(hours=12)

        preDeadLineDateStr = preDeadLineDate.strftime("%m%d%H")
        postDeadLineDateStr = postDeadLineDate.strftime("%m%d%H")

        preFileName = os.path.join(file_dir_name, "C1D" + mdhStr + "00" + preDeadLineDateStr + "001")
        postFileName = os.path.join(file_dir_name, "C1D" + mdhStr + "00" + postDeadLineDateStr + "001")

        fileNameList.append(preFileName)
        fileNameList.append(postFileName)

        preFileNameDict[postFileName] = preFileName

    return fileNameList, preFileNameDict


def searchMicapsFileByGribFileName(fileName):
    fileNameSplit = fileName.split("/")
    fileNameStr = fileNameSplit[-1]
    oldMdhStr = fileNameStr[-9:-3]
    oldYearStr = fileNameSplit[-3]

    labelDateStr = oldYearStr + oldMdhStr
    labelDate = datetime.datetime.strptime(labelDateStr, "%Y%m%d%H") + datetime.timedelta(hours=8)
    yearStr = labelDate.strftime("%Y")
    monthStr = labelDate.strftime("%m")

    labelDirName = os.path.join(micaps_path, yearStr, monthStr, "surface", "r6-p")
    labelFileName = datetime.datetime.strftime(labelDate, "%Y%m%d%H")[2:] + ".000"
    labelFileFullName = os.path.join(labelDirName, labelFileName)

    return labelFileFullName


def getOneDateMicapsDataList(micapsFileName):
    labels = []

    if not os.path.exists(micapsFileName):
        print(micapsFileName)
        return labels

    with open(micapsFileName, encoding="GBK") as f:
        data = f.readlines()
        label_data = data[14:]
        for oneLine in label_data:
            oneLabel = oneLine.split()
            index, lon, lat, _, values = map(float, oneLabel)
            if (lat < 28) or (lat > 36):
                continue
            elif (lon < 112) or (lon > 124):
                continue
            else:
                target = (index, lat, lon, values)
                labels.append(target)
    return labels


def cropGribByMicapsData(micapsDataList, normalGribs, precipitationGribs, preGribData=None):
    featureKeysList = [grib.name + "_level_" + str(grib.level) for grib in normalGribs]
    featureKeysList += [grib.name for grib in precipitationGribs]
    # indexList = [item[0] for item in micapsDataList]

    xrDict = {}


    for index, lat, lon, value in tqdm.tqdm(
            micapsDataList, total=len(micapsDataList),
            desc='Micaps Data Loading ', ncols=80,
            leave=False):

        latLowerBound = math.floor(lat * 4) / 4 - 1
        lonLowerBound = math.floor(lon * 4) / 4 - 1
        latUpperBound = latLowerBound + 2
        lonUpperBound = lonLowerBound + 2

        featureValuesList = []
        for grib in normalGribs:
            featureValues = grib.data(lat1=latLowerBound, lat2=latUpperBound, lon1=lonLowerBound, lon2=lonUpperBound)[0]
            if featureValues.shape[0] < len(largeAxis):
                featureValues = interpolate.interp2d(smallAxis, smallAxis, featureValues, kind='linear')(largeAxis,
                                                                                                         largeAxis)
            featureValuesList.append(featureValues)

        latLowerBound = math.floor(lat * 8) / 8 - 1
        lonLowerBound = math.floor(lon * 8) / 8 - 1
        latUpperBound = latLowerBound + 2.01
        lonUpperBound = lonLowerBound + 2.01

        latAxis = np.arange(latLowerBound, latUpperBound, 0.125)
        lonAxis = np.arange(lonLowerBound, lonUpperBound, 0.125)

        precipitationValuesList = [
            grib.data(lat1=latLowerBound, lat2=latUpperBound, lon1=lonLowerBound, lon2=lonUpperBound)[0] for grib in
            precipitationGribs]
        if preGribData is not None:
            prePrecipitationValuesList = [
                grib.data(lat1=latLowerBound, lat2=latUpperBound, lon1=lonLowerBound, lon2=lonUpperBound)[0] for grib in
                preGribData]
            precipitationValuesList = list(
                map(lambda x: x[1] - x[0], zip(prePrecipitationValuesList, precipitationValuesList)))

        featureValuesList += precipitationValuesList

        featureValues = np.stack(featureValuesList)

        foo = xr.DataArray(featureValues, coords=[featureKeysList, latAxis, lonAxis], dims=['Key', 'Lats', 'Lons'])
        foo.attrs["micapValues"] = value

        xrDict[index] = foo
        # print(foo)
        #
        # concatFoo = xr.concat([foo, foo], pd.Index([1, 2], name='index'))
        # print(concatFoo[0])
        # print(concatFoo[0].values.shape)
        # break

    return xrDict


if __name__ == "__main__":
    dateList = ["20160801", "20160901", "20161001", "20161101", "20161201",
                "20170401", "20170501", "20170601", "20170701", "20170801", "20170901", "20171001"]
    dataPreLoader(sys.argv[1])
    #
    # fileName = "2016080106.pkl"
    # with open(fileName,'rb') as f:
    #     data = pickle.load(f)
    #
    # keysList = list(data.keys())
    # print(data[keysList[0]])