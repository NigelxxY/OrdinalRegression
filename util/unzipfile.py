import os
import datetime
import subprocess

root_path = "/mnt/pami/DATASET/METEOROLOGY/ECforecast/GRIB"
start_date = datetime.datetime.strptime("20160731", "%Y%m%d")
end_date = datetime.datetime.strptime("20171231", "%Y%m%d")
# fileNameList = []
date_list = [start_date + datetime.timedelta(hours=12 * x) for x in
             range(0, 2 * (end_date - start_date).days)]
for oneDate in date_list:
    yearStr = oneDate.strftime("%Y")
    mdhStr = oneDate.strftime("%m%d%H")
    file_dir_name = os.path.join(root_path, yearStr, mdhStr)
    if not os.path.exists(file_dir_name):
        continue

    deadlineDateList = [oneDate + datetime.timedelta(hours=6 * i) for i in range(1, 3)]
    for deadlineDate in deadlineDateList:
        deadlineDataStr = deadlineDate.strftime("%m%d%H")
        oneFileName = "C1D" + mdhStr + "00" + deadlineDataStr + "001"
        oneFileFullName = os.path.join(file_dir_name, oneFileName)
        if os.path.exists(oneFileFullName):
            print(oneFileName, " Exists")
            continue
        elif os.path.exists(oneFileFullName + '.bz2'):
            p = subprocess.Popen(['sudo', 'bunzip2', '-v', oneFileFullName + '.bz2'], stdout=subprocess.PIPE)
            output = p.stdout.read().decode("utf-8").split()
            print(output)
        else:
            print("NO FILENAME EXISTS ", oneFileName)

            # fileNameList.append(os.path.join(file_dir_name, oneFileName))
