#!usr/bin/python
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.figure(figsize=(8,4))

with open("/Users/zengly/Downloads/P2PNetwork/test/sim_output_origin.csv", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
xx = data[1].split(",")[3:-1]
yy = data[2].split(",")[3:-1]
for k in xx:
	x_.append(float(k))
for k in yy:
	y_.append(float(k))
avg_ = data[7].replace("\n","")
print avg_
plt.plot(x_, y_,marker="o",label='normal(%s)'%avg_)


with open("/Users/zengly/Downloads/P2PNetwork/test/sim_output_inflation5.csv", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
xx = data[1].split(",")[3:-1]
yy = data[2].split(",")[3:-1]
for k in xx:
	x_.append(float(k))
for k in yy:
	y_.append(float(k))
avg_ = data[7].replace("\n","").split("=")[1].replace(" ", "")
print avg_
plt.plot(x_, y_,marker="x",label="inflation attack 5%%(%s)"%avg_)

with open("/Users/zengly/Downloads/P2PNetwork/test/sim_output_inflation10.csv", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
xx = data[1].split(",")[3:-1]
yy = data[2].split(",")[3:-1]
for k in xx:
	x_.append(float(k))
for k in yy:
	y_.append(float(k))
avg_ = data[7].replace("\n","").split("=")[1].replace(" ", "")
print avg_
plt.plot(x_, y_,marker="x",label="inflation attack 10%%(%s)"%avg_)

with open("/Users/zengly/Downloads/P2PNetwork/test/sim_output_inflation20.csv", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
xx = data[1].split(",")[3:-1]
yy = data[2].split(",")[3:-1]
for k in xx:
	x_.append(float(k))
for k in yy:
	y_.append(float(k))
avg_ = data[7].replace("\n","").split("=")[1].replace(" ", "")
print avg_
plt.plot(x_, y_,marker="x",label="inflation attack 20%%(%s)"%avg_)

with open("/Users/zengly/Downloads/P2PNetwork/test/sim_output_inflation30.csv", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
xx = data[1].split(",")[3:-1]
yy = data[2].split(",")[3:-1]
for k in xx:
	x_.append(float(k))
for k in yy:
	y_.append(float(k))
avg_ = data[7].replace("\n","").split("=")[1].replace(" ", "")
print avg_
plt.plot(x_, y_,marker="x",label="inflation attack 30%%(%s)"%avg_)

with open("/Users/zengly/Downloads/P2PNetwork/test/sim_output_inflation40.csv", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
xx = data[1].split(",")[3:-1]
yy = data[2].split(",")[3:-1]
for k in xx:
	x_.append(float(k))
for k in yy:
	y_.append(float(k))
avg_ = data[7].replace("\n","").split("=")[1].replace(" ", "")
print avg_
plt.plot(x_, y_,marker="x",label="inflation attack 40%%(%s)"%avg_)

with open("/Users/zengly/Downloads/P2PNetwork/test/sim_output_inflation50.csv", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
xx = data[1].split(",")[3:-1]
yy = data[2].split(",")[3:-1]
for k in xx:
	x_.append(float(k))
for k in yy:
	y_.append(float(k))
avg_ = data[7].replace("\n","").split("=")[1].replace(" ", "")
print avg_
plt.plot(x_, y_,marker="x",label="inflation attack 50%%(%s)"%avg_)

#####
with open("/Users/zengly/Downloads/P2PNetwork/test/sim_output_random.csv", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
xx = data[1].split(",")[3:-1]
yy = data[2].split(",")[3:-1]
for k in xx:
	x_.append(float(k))
for k in yy:
	y_.append(float(k))
avg_ = data[7].replace("\n","").split("=")[1].replace(" ", "")
print avg_
plt.plot(x_, y_,marker="s",label="random(%s)"%avg_)
#####

import matplotlib
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
xmajorLocator   = MultipleLocator(0.1)
xminorLocator   = MultipleLocator(0.05)
plt.gca().xaxis.set_major_locator(xmajorLocator)
plt.gca().xaxis.set_minor_locator(xminorLocator)


plt.xlim([0, 1])
plt.ylabel("Latency PDF")
#plt.title("Placement under 30% inflation")
plt.legend(ncol=2, loc='upper left',fontsize=7.5,fancybox=True,shadow=True,)
#plt.legend(bbox_to_anchor=(0., 1.01, 1., .101), loc=3,
#           ncol=3, mode="expand", borderaxespad=0., fontsize=14, markerscale=1.8)


plt.savefig('/Users/zengly/Downloads/P2PNetwork/test/graphs/broadcast_inflation_attack.pdf',bbox_inches = 'tight')
plt.show()

