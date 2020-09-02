# -*- coding: UTF-8 -*-
import math
from pyproj import CRS
from pyproj import Transformer
from pyproj import _datadir, datadir  # 加上这个linux打包执行不报错
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap

crs_WGS84 = CRS.from_epsg(4326)  # WGS84地理坐标系
crs_WebMercator = CRS.from_epsg(3857)  # Web墨卡托投影坐标系
cell_size = 0.009330691929342804  # 分辨率（米），一个像素表示的大小(24级瓦片)
origin_level = 24  # 原始瓦片级别
EarthRadius = 6378137.0  # 地球半径
tile_size = 256  # 瓦片大小

# describe a tree
adj = [[] for i in range(3000)]
fa = [0] * 3000
depth = [0] * 3000
col = ['r', 'g', 'b', 'c', 'y', 'k', 'orange', 'purple'] * 2
marker = ['o', 'v', '^', '<', '>', '+', 'D', '.', '1', '2'] * 2


def GK2WGS84(x_y, d):
    """
    高斯坐标转WGS84坐标
    :param x_y: 高斯坐标x,y集合
    :param d: 带号
    :return: 纬度,经度集合
    """
    format = '+proj=tmerc +lat_0=0 +lon_0=' + str(d * 3) + ' +k=1 +x_0=500000 +y_0=0 +ellps=WGS84 +units=m +no_defs'
    crs_GK = CRS.from_proj4(format)
    transformer = Transformer.from_crs(crs_GK, crs_WGS84)
    lat_lon = transformer.itransform(x_y)
    return lat_lon


def WGS84ToWebMercator(lat_lon):
    """
    WGS84坐标转web墨卡托坐标
    :param lat_lon:  纬度,经度集合
    :return:  web墨卡托坐标x,y集合
    """
    transformer = Transformer.from_crs(crs_WGS84, crs_WebMercator)
    x_y = transformer.itransform(lat_lon)
    return x_y


def WebMercator2WGS84(x_y):
    """
    web墨卡托坐标转WGS84坐标
    :param x_y:  web墨卡托坐标x,y集合
    :return:  纬度,经度集合
    """
    transformer = Transformer.from_crs(crs_WebMercator, crs_WGS84)
    lat_lon = transformer.itransform(x_y)
    return lat_lon


def GK2WGS84_Single(x, y, d):
    """
    高斯坐标转WGS84坐标
    :param x: 高斯坐标x
    :param y: 高斯坐标y
    :param d: 带号
    :return: 纬度,经度
    """
    format = '+proj=tmerc +lat_0=0 +lon_0=' + str(d * 3) + ' +k=1 +x_0=500000 +y_0=0 +ellps=WGS84 +units=m +no_defs'
    crs_GK = CRS.from_proj4(format)
    transformer = Transformer.from_crs(crs_GK, crs_WGS84)
    lat, lon = transformer.transform(x, y)
    return lat, lon


def WGS84ToGK_Single(lat, lon):
    """
    WGS84坐标转高斯坐标
    :param lat:  WGS84坐标纬度
    :param lon:  WGS84坐标经度
    :return: 高斯坐标x,y
    """
    d = int((lon + 1.5) / 3)
    format = '+proj=tmerc +lat_0=0 +lon_0=' + str(d * 3) + ' +k=1 +x_0=500000 +y_0=0 +ellps=WGS84 +units=m +no_defs'
    crs_GK = CRS.from_proj4(format)
    transformer = Transformer.from_crs(crs_WGS84, crs_GK)
    x, y = transformer.transform(lat, lon)
    return x, y


def WGS84ToWebMercator_Single(lat, lon):
    """
    WGS84坐标转web墨卡托坐标
    :param lat:  WGS84坐标纬度
    :param lon:  WGS84坐标经度
    :return:  web墨卡托坐标x,y
    """
    transformer = Transformer.from_crs(crs_WGS84, crs_WebMercator)
    x, y = transformer.transform(lat, lon)
    return x, y


def WebMercator2WGS84_Single(x, y):
    """
    web墨卡托坐标转WGS84坐标
    :param x:  web墨卡托坐标x
    :param y:  web墨卡托坐标y
    :return:  纬度,经度
    """
    transformer = Transformer.from_crs(crs_WebMercator, crs_WGS84)
    lat, lon = transformer.transform(x, y)
    return lat, lon


def pixel2WebMercator(pixel, min_x, min_y, height, cell_size):
    """
    像素坐标转web墨卡托坐标
    :param pixel: 像素坐标
    :param min_x: web墨卡托坐标的最小x值
    :param min_y: web墨卡托坐标的最小y值
    :param height: 图片高
    :param cell_size: 地面分辨率（1像素代表多少米）
    :return: web墨卡托坐标
    """
    x = pixel[0] * cell_size + min_x
    y = (height - pixel[1]) * cell_size + min_y
    return x, y


def GK2WebMercator_Single(x, y, d):
    """
    高斯坐标转Web墨卡托坐标
    :param x: 高斯坐标x值
    :param y: 高斯坐标y值
    :param d: 带号
    :return: web墨卡托坐标
    """
    format = '+proj=tmerc +lat_0=0 +lon_0=' + str(d * 3) + ' +k=1 +x_0=500000 +y_0=0 +ellps=WGS84 +units=m +no_defs'
    crs_GK = CRS.from_proj4(format)
    transformer = Transformer.from_crs(crs_GK, crs_WebMercator)
    web_x, web_y = transformer.transform(x, y)
    return web_x, web_y


def web2Pixel(x, y, level):
    """
    Web墨卡托坐标转像素坐标
    :param x: Web墨卡托坐标x值
    :param y: Web墨卡托坐标y值
    :param level: 瓦片级别
    :return: 像素坐标
    """
    real_cell_size = cell_size * math.pow(2, (origin_level - level))
    pixel_x = math.floor((x + math.pi * EarthRadius) / real_cell_size)
    pixel_y = math.floor((math.pi * EarthRadius - y) / real_cell_size)
    return pixel_x, pixel_y


def pixel2Web(pixel_x, pixel_y, level):
    """
    像素坐标转Web墨卡托坐标
    :param pixel_x: 像素x坐标
    :param pixel_y: 像素y坐标
    :param level: 瓦片级别
    :return: web墨卡托坐标
    """
    real_cell_size = cell_size * math.pow(2, (origin_level - level))
    web_x = pixel_x * real_cell_size - (math.pi * EarthRadius)
    web_y = math.pi * EarthRadius - (pixel_y * real_cell_size)
    return web_x, web_y


def pixelGetTile(pixel):
    """
    计算像素所在的瓦片号
    :param pixel: 像素坐标
    :return: 瓦片行列号
    """
    tile_x = int(pixel[0] / tile_size)
    tile_y = int(pixel[1] / tile_size)
    return tile_x, tile_y

def rad(x) :
    pi = np.arccos(-1)
    print(pi)
    return x * pi / 180.0

def distance(latA, lonA, latB, lonB):
    latA = rad(latA)
    latB = rad(latB)
    lonA = rad(lonA)
    lonB = rad(lonB)
    C = np.cos(latA) * np.cos(latB) * np.cos(lonA - lonB) + np.sin(latA) * np.sin(latB)
    R = 6371000
    AC = np.arccos(C)
    print(AC)
    return R * AC

def dfs(x):
    for y in adj[x]:
        depth[y] = depth[x] + 1
        dfs(y)

def draw_tree() :
    input = open("tree_struct.txt", "r")
    first_line = input.readline()
    n, root = map(int, first_line.strip().split(' '))

    for i in range(n):
        line = input.readline()
        print(i, line)
        parent = int(line.strip())
        fa[i] = parent
        if (parent != -1 and i != root):
            adj[parent].append(i)

    print(adj[0])
    dfs(root)
    print(depth)

    for i in range(n):
        parent = fa[i]
        #plt.scatter(data[i][1], data[i][0], marker = marker[depth[i]])
        #plt.scatter(data[i][1], data[i][0], marker = marker[depth[i]])
        if (parent != -1 and depth[parent] < 100):
            x = [data[parent][1], data[i][1]]
            y = [data[parent][0], data[i][0]]
            #plt.plot(x, y, color = col[depth[parent]])
            plt.arrow(x[0], y[0], x[1] - x[0], y[1] - y[0], color = col[depth[parent]], head_width = 1.2)
            plt.scatter(data[i][1], data[i][0], marker = marker[depth[i]])


input = open("geolocation.csv", "r")
data = np.zeros((10000, 10), dtype=np.float)
i = 0

line = input.readline()
while line:
    #print(i, line, end = "")
    cur = line.split(',')
    for j in range(len(cur)) :
        data[i][j] = float(cur[j])
    line = input.readline()
    i += 1
n = i

'''
print(data[0])
print(data[4])
dist = distance(data[0][0], data[0][1], data[4][0], data[4][1])
print(dist)
print(dist / (2.0 * (10 ** 8)))
exit()
'''

'''
m = Basemap(projection='cyl', resolution=None,
            llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, )

plt.show()
plt.savefig("map.pdf")
exit()
'''

#output = open("geolocation.txt", "w")

#print(n, file = output)

x_arr = []
y_arr = []

for i in range(n):
    #print(data[i][0], data[i][1])
    #x, y = WGS84ToGK_Single(data[i][0], data[i][1])
    x, y = data[i][0], data[i][1]
    #print(x, y, file = output)
    #print(x, y)
    x_arr.append(x)
    y_arr.append(y)
#exit()

#plt.scatter(x_arr, y_arr, )

#plt.scatter(y_arr[0:100], x_arr[0:100])
draw_tree()
plt.savefig("map.pdf")
plt.show()




