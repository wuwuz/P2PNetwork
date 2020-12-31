#!usr/bin/python
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.figure(figsize=(8,4))

#err_median_64.txt basic ce=0.25 cc=0.25 fix
#err_median_planet.txt
#err_median_newton_shift25.txt

"""
with open("/Users/zengly/Downloads/P2PNetwork/test/outputs/err_median_newton_shift25.txt", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
i = 1
for k in data:
    print (k)
    x_.append(i)
    i+=1
    y_.append(float(k.replace("\n", "")))
plt.plot(x_, y_, linewidth=2,label="newton(static)")

"""


with open("/Users/zengly/Downloads/P2PNetwork/test/outputs/coordinate_origin.txt", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
for k in data:
    print (k)
    li = k.split(',')
    x_.append(float(li[0]))
    y_.append(float(li[1]))
plt.scatter(x_, y_,alpha=0.5,label='origin')#label='origin'



with open("/Users/zengly/Downloads/P2PNetwork/test/outputs/coordinate_breakdown_30.txt", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
for k in data:
    print (k)
    li = k.split(',')
    x_.append(float(li[0]))
    y_.append(float(li[1]))
plt.scatter(x_, y_,alpha=0.5,label="breakdown")#label="breakdown"

with open("/Users/zengly/Downloads/P2PNetwork/test/outputs/coordinate_disorder_30.txt", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
for k in data:
    print (k)
    li = k.split(',')
    x_.append(float(li[0]))
    y_.append(float(li[1]))
#plt.scatter(x_, y_,alpha=0.5, )#label="disorder"

with open("/Users/zengly/Downloads/P2PNetwork/test/outputs/coordinate_deflation_30.txt", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
for k in data:
    print (k)
    li = k.split(',')
    x_.append(float(li[0]))
    y_.append(float(li[1]))
plt.scatter(x_, y_,alpha=0.5, label="deflation")#label="deflation"

with open("/Users/zengly/Downloads/P2PNetwork/test/outputs/coordinate_inflation_30.txt", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
for k in data:
    print (k)
    li = k.split(',')
    x_.append(float(li[0]))
    y_.append(float(li[1]))
#plt.scatter(x_, y_,alpha=0.5, ) #label="inflation"

plt.xlabel("X")
plt.ylabel("Y")
#plt.title("Placement under 30% inflation")
#plt.legend(loc='upper right', fancybox=True,shadow=True,)
plt.legend(bbox_to_anchor=(0., 1.01, 1., .101), loc=3,
           ncol=3, mode="expand", borderaxespad=0., fontsize=14, markerscale=1.8)

"""
plt.title("Inflation Attack(malicious_cor=(800,800))")
plt.xlabel("Round")
plt.ylabel("Prediction Error(ms)")
#plt.yscale("log")
plt.legend(loc='upper right', fancybox=True,shadow=True,)
"""

plt.savefig('/Users/zengly/Downloads/P2PNetwork/test/graphs/placement_newtion_attacks.pdf',bbox_inches = 'tight')
plt.show()



