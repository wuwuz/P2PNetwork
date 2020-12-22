#!usr/bin/python
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.figure(figsize=(8,4))

#err_median_64.txt basic ce=0.25 cc=0.25 fix


with open("/Users/zengly/Downloads/P2PNetwork/test/err_median_planet.txt", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
i = 1
for k in data:
    print (k)
    x_.append(i)
    i+=1
    y_.append(float(k.replace("\n", "")))
plt.plot(x_, y_, linewidth=2,label="random 64 median")

with open("/Users/zengly/Downloads/P2PNetwork/test/err_mean_planet.txt", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
i = 1
for k in data:
    print (k)
    x_.append(i)
    i+=1
    y_.append(float(k.replace("\n", "")))
plt.plot(x_, y_, linewidth=2,label="random 64 mean")


with open("/Users/zengly/Downloads/P2PNetwork/test/err_median_planet_close.txt", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
i = 1
for k in data:
    print (k)
    x_.append(i)
    i+=1
    y_.append(float(k.replace("\n", "")))
plt.plot(x_, y_, linewidth=2,label="random/close 64 median")

with open("/Users/zengly/Downloads/P2PNetwork/test/err_mean_planet_close.txt", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
i = 1
for k in data:
    print (k)
    x_.append(i)
    i+=1
    y_.append(float(k.replace("\n", "")))
plt.plot(x_, y_, linewidth=2,label="random/close 64 mean")


plt.xlabel("Round")
plt.ylabel("Prediction Error(ms)")
plt.legend(loc='upper right', fancybox=True,shadow=True,)

plt.savefig('/Users/zengly/Downloads/P2PNetwork/test/vivaldi_planet_100_64.pdf',bbox_inches = 'tight')
plt.show()
