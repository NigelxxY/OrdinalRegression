import pygrib
import os
import json
import datetime
import tqdm
import math
import torch
import numpy as np
from torch.utils import data
import torch.nn.functional as F


def raw_data_loader():
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



def collate_fn(batch):
    """
    Check input grib values less than zero


    :param batch: a list of (grib,label)
    :return: data: a torch tensor of shape (batch_size,1,grib_len,grib_width)
             label: a torch tensor of shape (batch_size, 1)
    """
    gribData,labels = zip(*batch)
    for idx,item in enumerate(gribData):
        gribData[idx] = item*(item>0)

    gribData = torch.stack(gribData,0)
    labels = torch.stack(labels,0)
    return gribData,labels

"""
为了解决样本不均衡的问题
利用了一个multinomial把权重加上了
"""
class gribDataSampler4NormalSlice(data.sampler.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.weight = torch.Tensor([1 if label == 0 else 4 for data,label in self.data_source],dtype=torch.double)
        self.num_sample = 30000
        self.replacement = True

    def __iter__(self):
        return iter(torch.multinomial(self.weight,self.num_sample,self.replacement))

    def __len__(self):
        return self.num_sample


class gribDataSet4NormalSlice(data.Dataset):
    def __init__(self, args, train=True):
        # if isinstance(config_file, str):
        #     with open(config_file, 'r') as f:
        #         config_file = json.load(f)
        # if not isinstance(config_file, dict):
        #     raise NotImplementedError

        self.root_path = args.root_path
        self.micaps_path = args.micaps_path
        if train:
            self.start_date = datetime.datetime.strptime(args.start_date, "%Y%m%d")
            self.end_date = datetime.datetime.strptime(args.end_date, "%Y%m%d")
        else:
            self.start_date = datetime.datetime.strptime(args.end_date, "%Y%m%d")
            self.end_date = self.start_date + datetime.timedelta(days=10)
        print(self.start_date)
        print(self.end_date)
        self.data = {}
        self.predata = {}
        self.datalen = 0
        self.i = 0
        self.generate_data()

    def __len__(self):
        return self.datalen-1

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

    def __next__(self):
        if self.i < self.datalen:
            data = self.data[self.i][0]
            label = self.data[self.i][1]
            self.i += 1
            return data,label
        else:
            self.i = 0
            raise StopIteration

    def __iter__(self):
        return self
    """
    生成数据的核心函数：
    首先生成文件名，然后根据文件名去生成观测数据的
    然后根据观测数据去切网格数据
    然后将所有切过的数据缓存在self.data[index] = (gribData,labels)
    """
    def generate_data(self):
        self.fileNameList = self.generate_file_path()
        # print(self.fileNameList)
        for idx, fileName in tqdm.tqdm(
                enumerate(self.fileNameList), total=len(self.fileNameList),
                desc='Load Data', ncols=80,
                leave=False):
            gribData = pygrib.open(fileName)
            totalPreciptation = gribData.select(name="Total precipitation")[0]

            if fileName in self.predata.keys():
                preGribData = pygrib.open(fileName)
                preTotalPreciptation = preGribData.select(name="Total precipitation")[0]
            else:
                preTotalPreciptation = None

            labels = self.getLabel(fileName)
            self.slice_grib_by_label(totalPreciptation, labels, preTotalPreciptation)

    def slice_grib_by_label(self, totalPreciptation, labels, preTotalPreciptation=None):
        for (lat, lon, p_values) in tqdm.tqdm(
                labels, total=len(labels),
                desc='Load One Day Data', ncols=80,
                leave=False):
            lat_bottom = math.floor(lat*8)/8 - 1
            lon_bottom = math.floor(lon*8)/8 - 1
            lat_top = lat_bottom + 2
            lon_top = lon_bottom + 2
            numpyData, _, _ = totalPreciptation.data(lat1=lat_bottom, lat2=lat_top, lon1=lon_bottom, lon2=lon_top)
            numpyData = numpyData * (numpyData > 1e-4)
            if preTotalPreciptation is not None:
                preNumpyData, _, _ = preTotalPreciptation.data(lat1=lat_bottom, lat2=lat_top, lon1=lon_bottom,
                                                               lon2=lon_top)
                preNumpyData = preNumpyData * (preNumpyData > 1e-4)
                numpyData = numpyData - preNumpyData

            self.data[self.datalen] = (numpyData[np.newaxis, :].astype(float), np.array([p_values], dtype=float))
            self.datalen += 1

    def generate_file_path(self):
        fileNameList = []
        date_list = [self.start_date + datetime.timedelta(hours=12 * x) for x in
                     range(0, 2 * (self.end_date - self.start_date).days)]
        for oneDate in date_list:
            yearStr = oneDate.strftime("%Y")
            mdhStr = oneDate.strftime("%m%d%H")
            file_dir_name = os.path.join(self.root_path, yearStr, mdhStr)
            if not os.path.exists(file_dir_name):
                continue

            deadlineDateList = [oneDate + datetime.timedelta(hours=6 * i) for i in range(1, 3)]
            for idx, deadlineDate in enumerate(deadlineDateList):
                deadlineDateStr = deadlineDate.strftime("%m%d%H")
                oneFileName = "C1D" + mdhStr + "00" + deadlineDateStr + "001"
                fullFileName = os.path.join(file_dir_name, oneFileName)
                if idx > 0:
                    preDeadLineDateStr = deadlineDateList[idx - 1].strftime("%m%d%H")
                    preFileName = "C1D" + mdhStr + "00" + preDeadLineDateStr + "001"
                    self.predata[fullFileName] = os.path.join(file_dir_name, preFileName)
                fileNameList.append(fullFileName)

        return fileNameList

    def getLabel(self, fileName):
        fileNameSplit = fileName.split("/")
        fileNameStr = fileNameSplit[-1]
        oldMdhStr = fileNameStr[-9:-3]
        oldYearStr = fileNameSplit[-3]

        labelDateStr = oldYearStr + oldMdhStr
        labelDate = datetime.datetime.strptime(labelDateStr, "%Y%m%d%H") + datetime.timedelta(hours=8)
        yearStr = labelDate.strftime("%Y")
        monthStr = labelDate.strftime("%m")

        labelDirName = os.path.join(self.micaps_path, yearStr, monthStr, "surface", "r6-p")
        labelFileName = datetime.datetime.strftime(labelDate, "%Y%m%d%H")[2:] + ".000"
        labelFileFullName = os.path.join(labelDirName, labelFileName)

        labels = []

        if not os.path.exists(labelFileFullName):
            print(labelFileFullName)
            return labels

        with open(labelFileFullName, encoding="GBK") as f:
            data = f.readlines()
            label_data = data[14:]
            for oneLine in label_data:
                oneLabel = oneLine.split()
                lon = float(oneLabel[1])
                lat = float(oneLabel[2])
                values = float(oneLabel[4])
                if (lat < 28) or (lat > 36):
                    continue
                elif (lon < 112) or (lon > 124):
                    continue
                else:
                    target = (lat, lon, values)
                    labels.append(target)
        return labels


''' Abondoned code, Do not remove them.
class gribDataSet4NormalCache(data.Dataset):
    def __init__(self, config_file):
        if isinstance(config_file, str):
            with open(config_file, 'r') as f:
                config_file = json.load(f)
        if not isinstance(config_file, dict):
            raise NotImplementedError

        self.root_path = config_file["root_path"]
        self.micaps_path = config_file["micaps_path"]
        self.start_date = datetime.datetime.strptime(config_file["start_date"], "%Y%m%d")
        self.end_date = datetime.datetime.strptime(config_file["end_date"], "%Y%m%d")
        self.data = self.load_data()

    def __len__(self):
        return len(self.fileNameList)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.data[idx][2], self.data[idx][3]

    def load_data(self):
        data_dict = {}
        self.fileNameList = self.generate_file_path()
        # self.pre_data = 0
        for idx, fileName in tqdm.tqdm(
                enumerate(self.fileNameList), total=len(self.fileNameList),
                desc='Load Data', ncols=80,
                leave=False):
            gribData = pygrib.open(fileName)
            totalPreciptation = gribData.select(name="Total precipitation")[0]
            data, lats, lons = self.generate_east_china_data(totalPreciptation)
            # if idx % 2 != 0:
            #     data = data - self.pre_data
            # self.pre_data = data
            labels = self.getLabel(fileName)
            data_dict[idx] = (data[np.newaxis, :].astype(float), lats, lons, labels)

        return data_dict

    def getLabel(self, fileName):
        fileNameSplit = fileName.split("/")
        fileName = fileNameSplit[-1]
        mdhStr = fileNameSplit[-2]
        yearStr = fileNameSplit[-3]

        labelDirName = os.path.join(self.micaps_path, yearStr, mdhStr[:2], "surface", "r6-p")

        if (fileName[-5:-3] == "06"):
            hourStr = "05"
        elif fileName[-5:-3] == "12":
            hourStr = "11"
        elif fileName[-5:-3] == '18':
            hourStr = '17'
        else:
            hourStr = '23'

        labelFileName = yearStr[-2:] + fileName[-9:-5] + hourStr + ".000"
        labelFileFullName = os.path.join(labelDirName, labelFileName)

        labels = []
        with open(labelFileFullName, encoding="GBK") as f:
            data = f.readlines()
            label_data = data[14:]
            for oneLine in label_data:
                oneLabel = oneLine.split()
                lon = float(oneLabel[1])
                lat = float(oneLabel[2])
                values = float(oneLabel[3])
                if (lat < 29) or (lat > 35):
                    continue
                elif (lon < 113) or (lon > 123):
                    continue
                else:
                    target = (lat, lon, values)
                    labels.append(target)
        return labels

    def generate_east_china_data(self, data):
        lats, lons = data.latlons()
        lats_mask = (lats.T[0] > 28) & (lats.T[0] < 36)
        lons_mask = (lons[0] > 112) & (lons[0] < 124)
        lats = lats.T[0][lats_mask]
        lons = lons[0][lons_mask]
        resNumpy = data.values[lats_mask].T[lons_mask].T
        return resNumpy, lats, lons

    def generate_file_path(self):
        fileNameList = []
        date_list = [self.start_date + datetime.timedelta(hours=12 * x) for x in
                     range(0, 2 * (self.end_date - self.start_date).days)]
        for oneDate in date_list:
            yearStr = oneDate.strftime("%Y")
            mdhStr = oneDate.strftime("%m%d%H")
            file_dir_name = os.path.join(self.root_path, yearStr, mdhStr)
            if not os.path.exists(file_dir_name):
                continue

            deadlineDateList = [oneDate + datetime.timedelta(hours=6 * i) for i in range(1, 2)]
            for deadlineDate in deadlineDateList:
                deadlineDataStr = deadlineDate.strftime("%m%d%H")
                oneFileName = "C1D" + mdhStr + "00" + deadlineDataStr + "001"
                fileNameList.append(os.path.join(file_dir_name, oneFileName))

        return fileNameList


class gribDataSet4Normal(data.Dataset):
    def __init__(self, config_file):
        if isinstance(config_file, str):
            with open(config_file, 'r') as f:
                config_file = json.load(f)
        if not isinstance(config_file, dict):
            raise NotImplementedError

        self.root_path = config_file["root_path"]
        self.micaps_path = config_file["micaps_path"]
        self.start_date = datetime.datetime.strptime(config_file["start_date"], "%Y%m%d")
        self.end_date = datetime.datetime.strptime(config_file["end_date"], "%Y%m%d")
        self.fileNameList = self.generate_file_path()
        self.lableList = self.getLabel()

    def __len__(self):
        return len(self.lableList)

    def __getitem__(self, idx):
        idx = idx
        toLoadFileName = self.fileNameList[idx]
        gribData = pygrib.open(toLoadFileName)
        totalPreciptation = gribData.select(name="Total precipitation")[0]
        data, lats, lons = self.generate_east_china_data(totalPreciptation)
        # if idx % 2 != 0:
        #     data = data - self.pre_data
        # self.pre_data = data
        labels = self.getLabel(toLoadFileName)
        return data[np.newaxis, :].astype(float), lats, lons, labels

    def getLabel(self):
        labelList = [label for fileName in self.fileNameList for label in self.getLabel(fileName)]

    def getLabel(self, fileName):
        fileNameSplit = fileName.split("/")
        fileName = fileNameSplit[-1]
        mdhStr = fileNameSplit[-2]
        yearStr = fileNameSplit[-3]

        labelDirName = os.path.join(self.micaps_path, yearStr, mdhStr[:2], "surface", "r6-p")

        if (fileName[-5:-3] == "06"):
            hourStr = "05"
        elif fileName[-5:-3] == "12":
            hourStr = "11"
        elif fileName[-5:-3] == '18':
            hourStr = '17'
        else:
            hourStr = '23'

        labelFileName = yearStr[-2:] + fileName[-9:-5] + hourStr + ".000"
        labelFileFullName = os.path.join(labelDirName, labelFileName)

        labels = []
        with open(labelFileFullName, encoding="GBK") as f:
            data = f.readlines()
            label_data = data[14:]
            for oneLine in label_data:
                oneLabel = oneLine.split()
                lon = float(oneLabel[1])
                lat = float(oneLabel[2])
                values = float(oneLabel[3])
                if (lat < 29) or (lat > 35):
                    continue
                elif (lon < 113) or (lon > 123):
                    continue
                else:
                    target = (lat, lon, values)
                    labels.append(target)
        return labels

    def generate_east_china_data(self, data):
        lats, lons = data.latlons()
        lats_mask = (lats.T[0] > 28) & (lats.T[0] < 36)
        lons_mask = (lons[0] > 112) & (lons[0] < 124)
        lats = lats.T[0][lats_mask]
        lons = lons[0][lons_mask]
        resNumpy = data.values[lats_mask].T[lons_mask].T
        return resNumpy, lats, lons

    def generate_file_path(self):
        fileNameList = []
        date_list = [self.start_date + datetime.timedelta(hours=12 * x) for x in
                     range(0, 2 * (self.end_date - self.start_date).days)]
        for oneDate in date_list:
            yearStr = oneDate.strftime("%Y")
            mdhStr = oneDate.strftime("%m%d%H")
            file_dir_name = os.path.join(self.root_path, yearStr, mdhStr)
            if not os.path.exists(file_dir_name):
                continue

            deadlineDateList = [oneDate + datetime.timedelta(hours=6 * i) for i in range(1, 2)]
            for deadlineDate in deadlineDateList:
                deadlineDataStr = deadlineDate.strftime("%m%d%H")
                oneFileName = "C1D" + mdhStr + "00" + deadlineDataStr + "001"
                fileNameList.append(os.path.join(file_dir_name, oneFileName))

        return fileNameList


class gribDataSet4Plum(data.Dataset):
    def __init__(self, config_file):
        self.root_path = config_file["root_path"]
        self.start_date = datetime.datetime.strptime()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def generaate_file_path(self):
        pass


class gribDataSet4Convective(data.Dataset):
    def __init__(self, config_file):
        self.root_path = config_file["root_path"]
        self.start_date = config_file["start_date"]
        self.end_date = config_file["end_date"]
        self.data = self.generate_file_path()

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def generate_file_path(self):
        start_date = datetime.datetime.strptime(self.start_date, "%Y%m%d")
        if (start_date < datetime.datetime(2016, 7, 31)):
            start_date = datetime.datetime(2016, 7, 31)
        end_date = datetime.datetime.strptime(self.end_date, "%Y%m%d")
        if (end_date >= datetime.datetime(2017, 11, 7)):
            end_date = datetime.datetime(2017, 11, 7)
        date_list = [start_date + datetime.timedelta(hours=12 * x) for x in range(0, 2 * (end_date - start_date).days)]
        for one_date in date_list:
            year_str = one_date.strftime("%Y")
            mdh_str = one_date.strftime("%m%d%H")
            grib_last_dir_name = os.path.join(self.root_path, year_str, mdh_str)
            if (os.path.exists(grib_last_dir_name)):
                pass
            else:
                continue
'''

'''
def parse_normal_data(config_data):
    root_path = config_data["root_path"]
    start_date = datetime.datetime.strptime(config_data["start_date"], "%Y%m%d")
    end_date = datetime.datetime.strptime(config_data["end_date"], "%Y%m%d")
    # split date with 12 hours
    date_list = [start_date + datetime.timedelta(hours=12 * x) for x in range(0, 2 * (end_date - start_date).days)]
    pprint(date_list)
    file_list = []
    no_date_start = datetime.datetime(2017, 1, 1)
    no_date_end = datetime.datetime(2017, 4, 1)
    for date in date_list:
        if ((date >= no_date_start) & & (date < no_date_end)):
            continue
        year_str = date.strftime("%Y")
        date_str = date.strftime("%m%d%H")
        file_dir_name = os.path.join(root_path, year_str, date_str)
        date_range_list = [date + datetime.timedelta(hours=3 * (i + 1)) for i in range(0, 4)]
        for date_range in date_range_list:
            data_range_str = date_range.strftime("%m%d%H")
            pprint(data_range_str)
            filename = "C1D" + date_str + "00" + data_range_str + "001"
            file_fullname = os.path.join(file_dir_name, filename)
            file_list.append(file_fullname)
            # pprint(os.path.exists(file_fullname))
    pprint(file_list)
    for idx, name in enumerate(file_list):
        if ((idx % 4) < 1):
            data = pygrib.open(name)
            total = data.select(name="Total precipitation")[0].values
            predata = total
            yield total
        else:
            data = pygrib.open(name)
            total = data.select(name="Total precipitation")[0].values
            res = total - predata
            predata = total
            yield res


def parse_plum_date(config_data):
    root_path = config_data["root_path"]
    start_date = datetime.datetime.strptime("20170622", "%Y%m%d")
    end_date = datetime.datetime.strptime("20170628", "%Y%m%d")
    # split date with 12 hours
    date_list = [start_date + datetime.timedelta(hours=12 * x) for x in range(0, 2 * (end_date - start_date).days)]
    pprint(date_list)
    file_list = []
    for date in date_list:
        year_str = date.strftime("%Y")
        date_str = date.strftime("%m%d%H")
        file_dir_name = os.path.join(root_path, year_str, date_str)
        date_range_list = [date + datetime.timedelta(hours=3 * (i + 1)) for i in range(0, 4)]
        for date_range in date_range_list:
            data_range_str = date_range.strftime("%m%d%H")
            pprint(data_range_str)
            filename = "C1D" + date_str + "00" + data_range_str + "001"
            file_fullname = os.path.join(file_dir_name, filename)
            file_list.append(file_fullname)
            # pprint(os.path.exists(file_fullname))
    pprint(file_list)
    for idx, name in enumerate(file_list):
        if ((idx % 4) < 1):
            data = pygrib.open(name)
            total = data.select(name="Total precipitation")[0]
            lats, lons = total.latlons()
            data = total.values
            lats_mask = (lats.T[0] > 25) & (lats.T[0] < 31)
            lats = lats.T[0][lats_mask]
            res_tmp = data[lats_mask]
            lons_mask = (lons[0] > 112) & (lons[0] < 124)
            res = res_tmp.T[lons_mask].T
            lons = lons[0][lons_mask]
            predata = total.values
            res_data = {"data": res,
                        "lats": lats,
                        "lons": lons,
                        "label": ""}
            yield res_data
        else:
            data = pygrib.open(name)
            total = data.select(name="Total precipitation")[0]
            lats, lons = total.latlons()
            data = total.values - predata
            lats_mask = (lats.T[0] > 25) & (lats.T[0] < 31)
            res_tmp = data[lats_mask]
            lats = lats.T[0][lats_mask]
            lons_mask = (lons[0] > 112) & (lons[0] < 124)
            lons = lons[0][lons_mask]
            res = res_tmp.T[lons_mask].T
            predata = total.values
            res_data = {"data": res,
                        "lats": lats,
                        "lons": lons,
                        "label": ""}
            yield res_data

    start_date = datetime.datetime.strptime("20170629", "%Y%m%d")
    end_date = datetime.datetime.strptime("20170706", "%Y%m%d")
    # split date with 12 hours
    date_list = [start_date + datetime.timedelta(hours=12 * x) for x in range(0, 2 * (end_date - start_date).days)]
    pprint(date_list)
    file_list = []
    for date in date_list:
        year_str = date.strftime("%Y")
        date_str = date.strftime("%m%d%H")
        file_dir_name = os.path.join(root_path, year_str, date_str)
        date_range_list = [date + datetime.timedelta(hours=3 * (i + 1)) for i in range(0, 4)]
        for date_range in date_range_list:
            data_range_str = date_range.strftime("%m%d%H")
            pprint(data_range_str)
            filename = "C1D" + date_str + "00" + data_range_str + "001"
            file_fullname = os.path.join(file_dir_name, filename)
            file_list.append(file_fullname)
            # pprint(os.path.exists(file_fullname))
    pprint(file_list)
    for idx, name in enumerate(file_list):
        if ((idx % 4) < 1):
            data = pygrib.open(name)
            total = data.select(name="Total precipitation")[0]
            lats, lons = total.latlons()
            data = total.values
            lats_mask = (lats.T[0] > 28) & (lats.T[0] < 36)
            res_tmp = data[lats_mask]
            lons_mask = (lons[0] > 112) & (lons[0] < 124)
            res = res_tmp.T[lons_mask].T
            predata = total.values
            lats = lats.T[0][lats_mask]
            lons = lons[0][lons_mask]
            res_data = {"data": res,
                        "lats": lats,
                        "lons": lons,
                        "label": ""}
            yield res_data
        else:
            data = pygrib.open(name)
            total = data.select(name="Total precipitation")[0]
            lats, lons = total.latlons()
            data = total.values - predata
            lats_mask = (lats.T[0] > 28) & (lats.T[0] < 36)
            res_tmp = data[lats_mask]
            lons_mask = (lons[0] > 112) & (lons[0] < 124)
            res = res_tmp.T[lons_mask].T
            predata = total.values
            lats = lats.T[0][lats_mask]
            lons = lons[0][lons_mask]
            res_data = {"data": res,
                        "lats": lats,
                        "lons": lons,
                        "label": ""}
            yield res_data


def parse_typhoon_data(config_data):
    root_path = config_data["root_path"]
    typhoon_filename_list = os.listdir(config_data["typhoon_path"])
    typhoon_filename_list = sorted(typhoon_filename_list)
    pprint(typhoon_filename_list)
    for typhoon_filename in typhoon_filename_list:
        filename = os.path.join(config_data["typhoon_path"], typhoon_filename)
        with open(filename, encoding="GBK") as f:
            data = f.readlines()
            for item in data:
                line = item.split()
                # pprint(line)
                if (line[7] == "16"):
                    year = "2016"
                else:
                    year = "2017"
                month = line[8]
                day = line[9]
                hour = line[10]
                if (int(hour) < 12):
                    date_str = month + day + "00"
                else:
                    date_str = month + day + "12"
                lon = float(line[12])
                lat = float(line[13])
                if ((lon < 110) or (lon > 124) or (lat < 20) or (lat > 40)):
                    continue
                lon_bottom = lon - 4;
                lon_top = lon + 4;
                lat_bottom = lat - 4;
                lat_top = lat + 4;
                filename_path = os.path.join(root_path, year, date_str)
                if (int(hour) < 3):
                    date_str_range = month + day + "03"
                    pre_flag = False
                elif (int(hour) < 6):
                    date_str_range = month + day + "06"
                    pre_flag = True
                    pre_data_str_range = month + day + "03"
                elif (int(hour) < 9):
                    date_str_range = month + day + "09"
                    pre_flag = True
                    pre_data_str_range = month + day + "06"
                else:
                    date_str_range = month + day + "12"
                    pre_flag = True
                    pre_data_str_range = month + day + "09"
                full_filename = os.path.join(filename_path, "C1D" + date_str + "00" + date_str_range + "001")
                pprint(full_filename)
                data = pygrib.open(full_filename).select(name="Total precipitation")[0]
                lats, lons = data.latlons()
                if (pre_flag):
                    pre_full_filename = os.path.join(filename_path,
                                                     "C1D" + date_str + "00" + pre_data_str_range + "001")
                    pre_data = pygrib.open(pre_full_filename).select(name="Total precipitation")[0]
                    data = data.values - pre_data.values
                lats_mask = (lats.T[0] > lat_bottom) & (lats.T[0] < lat_top)
                res_tmp = data[lats_mask]
                lons_mask = (lons[0] > lon_bottom) & (lons[0] < lon_top)
                res = res_tmp.T[lons_mask].T
                lats = lats.T[0][lats_mask]
                lons = lons[0][lons_mask]
                res_data = {"data": res,
                            "lats": lats,
                            "lons": lons,
                            "label": ""}
                yield res_data


def parse_convective_data(config_data):
    root_path = config_data["root_path"]
    start_date = datetime.datetime.strptime(config_data["start_date"], "%Y%m%d")
    end_date = datetime.datetime.strptime(config_data["end_date"], "%Y%m%d")
    # split date with 12 hours
    date_list = [start_date + datetime.timedelta(hours=12 * x) for x in range(0, 2 * (end_date - start_date).days)]
    pprint(date_list)
    file_list = []
    no_date_start = datetime.datetime(2017, 1, 1)
    no_date_end = datetime.datetime(2017, 4, 1)
    for date in date_list:
        if ((date >= no_date_start) & & (date < no_date_end)):
            continue
        year_str = date.strftime("%Y")
        date_str = date.strftime("%m%d%H")
        file_dir_name = os.path.join(root_path, year_str, date_str)
        date_range_list = [date + datetime.timedelta(hours=3 * (i + 1)) for i in range(0, 4)]
        for date_range in date_range_list:
            data_range_str = date_range.strftime("%m%d%H")
            pprint(data_range_str)
            filename = "C1D" + date_str + "00" + data_range_str + "001"
            file_fullname = os.path.join(file_dir_name, filename)
            file_list.append(file_fullname)
            # pprint(os.path.exists(file_fullname))
    pprint(file_list)
    for idx, name in enumerate(file_list):
        if ((idx % 4) < 1):
            data = pygrib.open(name)
            total = data.select(name="Total precipitation")[0]
            convective = data.select(name="Convective precipitation")[0]
            lats, lons = total.latlons()
            lats_mask = (lats.T[0] > 20) & (lats.T[0] < 40)
            lons_mask = (lons[0] > 110) & (lons[0] < 124)
            total_res = total.values[lats_mask].T[lons_mask].T
            convective_res = convective.values[lats_mask].T[lons_mask].T
            predata = total_res
            preconvective = convective_res
            lats = lats.T[0][lats_mask]
            lons = lons[0][lons_mask]
            if ((convective_res / total_res) > 0.8):
                res_data = {"data": total_res,
                            "lats": lats,
                            "lons": lons,
                            "label": ""}
                yield res_data
            else:
                continue
        else:
            data = pygrib.open(name)
            total = data.select(name="Total precipitation")[0]
            convective = data.select(name="Convective precipitation")[0]
            lats, lons = total.latlons()
            lats_mask = (lats.T[0] > 20) & (lats.T[0] < 40)
            lons_mask = (lons[0] > 110) & (lons[0] < 124)
            total_res = total.values[lats_mask].T[lons_mask].T
            convective_res = convective.values[lats_mask].T[lons_mask].T
            total_values = total_res - predata
            convective_values = convective_res - preconvective
            predata = total_res
            preconvective = convective_res
            lats = lats.T[0][lats_mask]
            lons = lons[0][lons_mask]
            if ((convective_values / total_values) > 0.8):
                res_data = {"data": total_values,
                            "lats": lats,
                            "lons": lons,
                            "label": ""}
                yield res_data
            else:
                continue


def data_loader(json_path):
    with open(json_path) as json_file:
        config_data = json.load(json_file)
    pprint(config_data)

    if (config_data["data_type"].lower() == "normal"):
        return parse_normal_data(config_data)
    elif (config_data["data_type"].lower() == "plum"):
        return parse_plum_date(config_data)
    elif (config_data["data_type"].lower() == "typhoon"):
        return parse_typhoon_data(config_data)
    elif (config_data["data_type"].lower() == "convection"):
        return parse_convective_data(config_data)
    else:
        pprint("ERROR!!!!")


if __name__ == "__main__":
    json_path = "./train_config.json"
    data_generater = data_loader(json_path)
    # pprint(next(data_generater).shape)
    # pprint(next(data_generater).shape)
    # pprint(next(data_generater))
'''
