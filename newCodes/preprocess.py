import os
import sys
import tqdm
import pygrib
import datetime
import numpy as np
from scipy import interpolate
from dateutil.relativedelta import relativedelta

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
                target = [index, lat, lon, values]
                labels.append(target)
    return labels

def datePreSaver(start_date):
    fileNameList, preFileNameDict = fileNameListGenerator(start_date)
    for fileName in tqdm.tqdm(
            fileNameList, total=len(fileNameList),
            desc='Load Data', ncols=80,
            leave=False):

        oneName = fileName.split('/')
        saveFileName = '_'.join([oneName[-3], oneName[-1][3:9], oneName[-1][11:17]]) + '.npy'
        labelFileName = searchMicapsFileByGribFileName(fileName)
        labels = np.array(getOneDateMicapsDataList(labelFileName))


        gribData = pygrib.open(fileName)
        preGribData = None
        if fileName in preFileNameDict.keys():
            preGribData = pygrib.open(preFileNameDict[fileName])

        normalData = [gribData.select(name=keyName, level=level)[0] for keyName in normalKeys for level in levelsList]
        normalData += [gribData.select(name=keyName)[0] for keyName in singleLevelKeys]
        precipitationData = [gribData.select(name=keyName)[0] for keyName in precipitationKeys]
        featureValuesList = []

        for grib in normalData:
            featureValues = \
                grib.data(lat1=locBound[0] - 1, lat2=locBound[1] + 1, lon1=locBound[2] - 1,
                          lon2=locBound[3] + 1)[0]
            if featureValues.shape[0] < len(largeLatAxis):
                featureValues = interpolate.interp2d(smallLonAxis, smallLatAxis, featureValues, kind='linear')(
                    largeLonAxis,
                    largeLatAxis)

            featureValuesList.append(featureValues)

        precipitationValuesList = [grib.data(lat1=locBound[0] - 1, lat2=locBound[1] + 1, lon1=locBound[2] - 1,
                                    lon2=locBound[3] + 1)[0] for grib in precipitationData]
        if preGribData is not None:
            prePrecipitationData = [preGribData.select(name=keyName)[0] for keyName in precipitationKeys]
            prePrecipitationValuesList = [grib.data(lat1=locBound[0] - 1, lat2=locBound[1] + 1, lon1=locBound[2] - 1,
                                    lon2=locBound[3] + 1)[0] for grib in prePrecipitationData]
            precipitationValuesList = list(
                map(lambda x: x[1] - x[0], zip(prePrecipitationValuesList, precipitationValuesList)))

        featureValuesList += precipitationValuesList
        features = np.stack(featureValuesList)

        np.save(os.path.join(features_save_path,saveFileName),features)
        np.save(os.path.join(labels_save_path,saveFileName),labels)


if __name__ == "__main__":
    datePreSaver(sys.argv[1])
