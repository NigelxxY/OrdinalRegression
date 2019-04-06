

import os
import tqdm
import datetime

start_dateStr = "20160801"
end_dateStr = "20171124"
root_path = "/mnt/pami/DATASET/METEOROLOGY/ECforecast/GRIB"
micaps_path = "/mnt/pami/DATASET/METEOROLOGY/GT_micaps"

start_date = datetime.datetime.strptime(start_dateStr, "%Y%m%d")
end_date = datetime.datetime.strptime(end_dateStr, "%Y%m%d")

def generate_file_path():
    fileNameList = []
    predata = {}
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
            if idx > 0:
                preDeadLineDateStr = deadlineDateList[idx - 1].strftime("%m%d%H")
                preFileName = "C1D" + mdhStr + "00" + preDeadLineDateStr + "001"
                predata[fullFileName] = os.path.join(file_dir_name, preFileName)
            fileNameList.append(fullFileName)

    return fileNameList


def getLabel(fileName):
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
        return []

    labels = []
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

if __name__ == "__main__":
    labels = []
    fileNameList = generate_file_path()
    zero_count = 0
    no_zero_count = 0
    for idx, fileName in tqdm.tqdm(
            enumerate(fileNameList), total=len(fileNameList),
            desc='Load Data', ncols=80,
            leave=False):
        labels+=getLabel(fileName)

    for idx, oneLabel in tqdm.tqdm(
            enumerate(labels), total=len(labels),
            desc='Statictis Labels', ncols=80,
            leave=False):
        # print(oneLabel)
        lat = oneLabel[0]
        lon = oneLabel[1]
        values = oneLabel[2]
        if values > 0:
            # print("Prep: %f, fileName is %s" % (values, fileName))
            # print("Lon is %f and Lat is %f" % (lon, lat))
            # print("============================================")
            no_zero_count += 1
        else:
            zero_count += 1

    print("Zero label nums is %d and No Zero Label Nums is %d" % (zero_count,no_zero_count))
