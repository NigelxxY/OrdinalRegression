import os
import json
import tqdm
import math
import pickle
import pygrib
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from dateutil.relativedelta import relativedelta
np.set_printoptions(suppress=True)

root_path = "/mnt/pami/DATASET/METEOROLOGY/ECforecast/GRIB"
micaps_path = "/mnt/pami/DATASET/METEOROLOGY/GT_micaps"
keys = ["Relative humidity",
        "Vertical velocity",
        "Temperature",
        "V component of wind",
        "U component of wind",
        "Potential vorticity",
        "Divergence",
        "Total column water",
        "Geopotential Height",
        "Large-scale precipitation",
        "Convective available potential energy",
        "Total cloud cover",
        "Total precipitation"]


# datetime.timedelta(month=1)


def fileNameListGenerator(start_date):
    fileNameList = []
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y%m%d")
    end_date = start_date + relativedelta(months=14)
    date_list = [start_date + datetime.timedelta(hours=12 * x) for x in
                 range(0, 2 * (end_date - start_date).days)]
    for oneDate in date_list:
        yearStr = oneDate.strftime("%Y")
        mdhStr = oneDate.strftime("%m%d%H")
        file_dir_name = os.path.join(root_path, yearStr, mdhStr)
        if not os.path.exists(file_dir_name):
            continue

        deadlineDateList = [oneDate + datetime.timedelta(hours=6 * i) for i in range(1, 3)]
        for idx, deadlineDate in enumerate(deadlineDateList):
            deadlineDateStr = deadlineDate.strftime("%m%d%H")
            oneFileName = "C1D" + mdhStr + "00" + deadlineDateStr + "001"
            fullFileName = os.path.join(file_dir_name, oneFileName)
            # if idx > 0:
            #     preDeadLineDateStr = deadlineDateList[idx - 1].strftime("%m%d%H")
            # preFileName = "C1D" + mdhStr + "00" + preDeadLineDateStr + "001"
            # self.predata[fullFileName] = os.path.join(file_dir_name, preFileName)
            fileNameList.append(fullFileName)
    return fileNameList


def micaps_multi_selector(fileNameList, micapsIndexList):
    labels = {}
    for idx, fileName in tqdm.tqdm(
            enumerate(fileNameList), total=len(fileNameList),
            desc=str(micaps_index) + ' Load Data', ncols=80,
            leave=False):
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

        if not os.path.exists(labelFileFullName):
            print(labelFileFullName)


def micaps_selector(fileNameList, micaps_index):
    labels = {}
    for idx, fileName in tqdm.tqdm(
            enumerate(fileNameList), total=len(fileNameList),
            desc=str(micaps_index) + ' Load Data', ncols=80,
            leave=False):
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

        p_values = 0

        if not os.path.exists(labelFileFullName):
            print(labelFileFullName)
        else:
            with open(labelFileFullName, encoding="GBK") as f:
                data = f.readlines()
                label_data = data[14:]
                for oneLine in label_data:
                    oneLabel = oneLine.split()
                    # lon = float(oneLabel[1])
                    # lat = float(oneLabel[2])
                    # values = float(oneLabel[4])
                    # if (lat < 28) or (lat > 36):
                    #     continue
                    # elif (lon < 112) or (lon > 124):
                    #     continue
                    # else:
                    #     target = (lat, lon, values)
                    #     labels.append(target)
                    if float(oneLabel[0]) == micaps_index:
                        p_values = float(oneLabel[4])
                        break
                        # labels.append(p_values)
        labels[labelDate] = p_values
    return labels
    # plt.switch_backend('agg')
    # plt.plot(labels.keys(),labels.values(),'r',linewidth=2)
    # # plt.show()
    # plt.xlabel("DateTime",fontsize=16)
    # plt.ylabel("P-Values",fontsize=16)
    # plt.savefig("./labelDisturbtion.jpg")
    # print(labels.values())


def feature_loader(fileNameList, lat, lon):
    load_data = {}
    grib_lat = math.floor(lat * 4) / 4
    grib_lon = math.floor(lon * 4) / 4

    for idx, fileName in tqdm.tqdm(
            enumerate(fileNameList), total=len(fileNameList),
            desc='Load Data', ncols=80,
            leave=False):
        gribData = pygrib.open(fileName)
        for key in keys:
            key_data_list = gribData.select(name=key)
            for key_data in key_data_list:
                dict_key = key + "_" + str(key_data.level)
                key_value = 0
                try:
                    key_array, _, _ = key_data.data(lat1=grib_lat, lat2=grib_lat, lon1=grib_lon, lon2=grib_lon)
                    key_value = key_array.item()
                except:
                    print(dict_key)
                if dict_key not in load_data.keys():
                    load_data[dict_key] = []
                load_data[dict_key].append(key_value)

    return load_data


def feature_saver(load_data, savePath):
    featureDataFrame = pd.DataFrame.from_dict(load_data)
    # savePath = "./" + saveFlag +".csv"
    featureDataFrame.to_csv(savePath)


def feature_corr_cal(micaps, date):
    fileNameList = fileNameListGenerator(date)
    savePath = os.path.join(os.getcwd(), str(micaps[0]), date) + ".csv"
    features = pd.DataFrame.from_csv(savePath)
    # print(features.columns)
    labels = pd.DataFrame(list(micaps_selector(fileNameList, micaps[0]).values()), columns=["labels"])
    features_labels = pd.concat([features, labels], axis=1)
    corrMatrix = features_labels.corr()
    return corrMatrix


def micapsListGenerator():
    returnList = []
    micaps_filename = "/mnt/pami/DATASET/METEOROLOGY/GT_micaps/2017/07/surface/r6-p/"
    filename_list = sorted(os.listdir(micaps_filename))
    item = filename_list[5]
    filename = os.path.join(micaps_filename, item)
    with open(filename, encoding="GBK") as f:
        data = f.readlines()
        micaps_list = data[14:]
        for items in micaps_list:
            micaps_data = items.split()
            lon = float(micaps_data[1])
            lat = float(micaps_data[2])

            if (lat < 28) or (lat > 36):
                continue
            elif (lon < 112) or (lon > 124):
                continue

            returnList.append(micaps_data)

    print(len(returnList))

    return returnList


def listMap(micapsList):
    return [list(map(float, item)) for item in micapsList]


def labelDistance(date):
    fileNameList = fileNameListGenerator(date)
    micapsList = listMap(micapsListGenerator())
    # print(micapsList[:10])
    labelsList = [micaps_selector(fileNameList, item[0]) for item in micapsList[:100]]

    distances = [calDistance(micapsList[0][2], micapsList[i][2], micapsList[0][1], micapsList[i][1]) for i in
                 range(1, 100)]
    corrs = [pearsonr(list(labelsList[0].values()), list(labelsList[i].values()))[0] for i in range(1, 100)]

    res = pd.DataFrame(data={"distances": distances, "corr": corrs})
    print(res)
    g = sns.pairplot(res)
    g.savefig("distance.png")


def heat_map_plot(corrMatrix):
    f, ax = plt.subplots(figsize=corrMatrix.shape)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corrMatrix,
                cmap=cmap,
                xticklabels=corrMatrix.columns.values,
                yticklabels=corrMatrix.columns.values,
                ax=ax)

    f.savefig("corrMatrix.jpg")


def linear_regression():
    pass


def calDistance(lat1, lat2, lon1, lon2):
    ra = 6378.140  # 赤道半径 (km)
    rb = 6356.755  # 极半径 (km)
    flatten = (ra - rb) / ra  # 地球扁率

    lat1, lat2, lon1, lon2 = map(math.radians, [lat1, lat2, lon1, lon2])

    pA = math.atan(rb / ra * math.tan(lat1))
    pB = math.atan(rb / ra * math.tan(lat2))
    xx = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(lon1 - lon2))
    c1 = (math.sin(xx) - xx) * (math.sin(pA) + math.sin(pB)) ** 2 / math.cos(xx / 2) ** 2
    c2 = (math.sin(xx) + xx) * (math.sin(pA) - math.sin(pB)) ** 2 / math.sin(xx / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (xx + dr)
    return distance


def featureMeanStdStatistics():
    cropRoot = "/mnt/pami/xyxu/dataset/cropGrib"
    cropFileList = [os.path.join(cropRoot, fileName) for fileName in os.listdir(cropRoot)]

    cacheList = []

    for fileName in cropFileList:
        with open(fileName, 'rb') as f:
            oneCropDataDict = pickle.load(f)
        for cropData in oneCropDataDict.values():
            features = cropData.values.reshape(37, 17 * 17)
            cacheList.append(features)

    concatenateFeatures = np.concatenate(cacheList, axis=1)
    return concatenateFeatures.mean(axis=1), concatenateFeatures.std(axis=1)
    # print(concatenateFeatures.shape)
    # print(concatenateFeatures.mean(axis=1))
    # print(concatenateFeatures.std(axis=1))
    # print(concatenateFeatures.min(axis=1))
    # print(concatenateFeatures.max(axis=1))


if __name__ == "__main__":
    featureMeanStdStatistics()
    # plt.switch_backend('agg')
    # micaps_one = list(map(float, ['57678', '112.55', '28.25', '75', '59.0']))
    # micaps_two = list(map(float, ['57687', '112.92', '28.22', '68', '120.0']))
    # dateList = ["20160801", "20160901", "20161001", "20161101", "20161201",
    #             "20170401", "20170501", "20170601", "20170701", "20170801", "20170901", "20171001"]
    # labelDistance(dateList[0])
    # corrMatrix = feature_corr_cal(micaps_one, dateList[2])
    # heat_map_plot(corrMatrix)
    # for date in dateList:
    #     fileNameList = fileNameListGenerator(date)
    #     feature_dict = feature_loader(fileNameList, lon=micaps_one[1], lat=micaps_one[2])
    #     dirPath = os.path.join(os.getcwd(),str(micaps_one[0]))
    #     if not os.path.exists(dirPath):
    #         os.mkdir(dirPath)
    #     savePath = os.path.join(dirPath,date)
    #     feature_saver(feature_dict,savePath=savePath)
    # labels_one = micaps_selector(fileNameList, micaps_one[0])
    # labels_two = micaps_selector(fileNameList, int(micaps_two[0]))
    # print(type(labels_one.values()))
    # print(pearsonr(list(labels_one.values()), list(labels_two.values())))
