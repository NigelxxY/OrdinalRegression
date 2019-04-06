import math
import torch.nn.functional as F

"""
将所有的对流天气的网格文件写入一个txt文件
需要足够的NAS空间
"""
def get_all_convective_filename():
    root_path = "/mnt/pami/DATASET/METEOROLOGY/ECforecast/GRIB"

"""
计算ts分数的，输入是正确，漏报和误报数量
返回值是ts分数
"""
def cal_ts(hits, misses, falsealarms):
    print("Hits nums {}, Miss nums {}, Falsealarm num {}".format(hits,misses,falsealarms))
    return hits / (hits + misses + falsealarms)

"""
对网格数据进行双线型插值的结果
输入是网格数据，网格的纬度列表，网格的经度列表，观测站的纬度，观测站的经度
返回是双线型插值的结果
"""
def torch_bilinear_interpolation(grib, grib_lats, grib_lons, gt_lat, gt_lon):
    lon_idx_0, lon_idx_1 = binary_search(grib_lons, gt_lon, False)
    lat_idx_0, lat_idx_1 = binary_search(grib_lats, gt_lat, True)
    print(lon_idx_0)
    print(lon_idx_1)
    print(lat_idx_0)
    print(lat_idx_1)
    # grib = grib[:,:,lat_idx_0:lat_idx_1,lon_idx_0:lon_idx_1]
    print(grib.shape)
    return F.interpolate(grib,size=[1,1],mode="bilinear").view(-1)


def grib_bilinear_interpolation(grib, grib_lats, grib_lons, gt_lat, gt_lon):
    lon_idx_0, lon_idx_1 = binary_search(grib_lons, gt_lon, False)
    lat_idx_0, lat_idx_1 = binary_search(grib_lats, gt_lat, True)
    # x_0 = grib_lons[lon_idx_0]
    # x_1 = grib_lons[lon_idx_1]
    # y_0 = grib_lats[lat_idx_0]
    # y_1 = grib_lats[lat_idx_1]
    # print(x_0,x_1,y_0,y_1)

    # w_0 = grib[:,:,lat_idx_1, lon_idx_0] * (x_1 - gt_lon) + grib[:,:,lat_idx_1, lon_idx_1] * (gt_lon - x_0 )
    # w_1 = grib[:,:,lat_idx_0, lon_idx_0] * (x_1 - gt_lon) / (x_1 - x_0 ) + grib[:,:,lat_idx_0, lon_idx_1] * (gt_lon - x_0 ) / (
    #     x_1 - x_0 )

    res = (grib[:,:,lat_idx_1, lon_idx_0] + grib[:,:,lat_idx_1, lon_idx_1] + grib[:,:,lat_idx_0, lon_idx_1] + grib[:,:,lat_idx_0, lon_idx_0])/4
    return res.view(-1)

"""
二分查找，返回离观测站最近的两个网格坐标点的索引
"""
def binary_search(list, item, is_lat):
    if is_lat:
        list=list[::-1]

    low = 0
    high = len(list) - 1
    while low <= high:
        mid = math.floor((low + high) / 2)
        guess = list[mid]
        if guess == item:
            if (is_lat):
                return len(list) - mid - 1, len(list) - mid
            else:
                return mid, mid+1
        elif guess > item:
            high = mid - 1
        else:
            low = mid + 1
    if (is_lat):
        return len(list) - low - 1, len(list) - high - 1
    else:
        return high, low