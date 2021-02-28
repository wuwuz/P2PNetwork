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
#avg_ = data[7].replace("\n","")
#print avg_
plt.plot(x_, y_,marker="o",label='BCast(Normal)')

with open("/Users/zengly/Downloads/P2PNetwork/test/sim_output_no_response_10.csv", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
xx = data[1].split(",")[3:-1]
yy = data[2].split(",")[3:-1]
for k in xx:
	x_.append(float(k))
for k in yy:
	y_.append(float(k))
plt.plot(x_, y_,marker="+",label='BCast(10% Malicious)')

with open("/Users/zengly/Downloads/P2PNetwork/test/sim_output_no_response_30.csv", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
xx = data[1].split(",")[3:-1]
yy = data[2].split(",")[3:-1]
for k in xx:
	x_.append(float(k))
for k in yy:
	y_.append(float(k))
plt.plot(x_, y_,marker="x",label='BCast(30% Malicious)')

with open("/Users/zengly/Downloads/P2PNetwork/test/sim_output_no_response_49.csv", "r") as f:
    data = f.readlines()
x_ = []
y_ = []
xx = data[1].split(",")[3:-1]
yy = data[2].split(",")[3:-1]
for k in xx:
	x_.append(float(k))
for k in yy:
	y_.append(float(k))
plt.plot(x_, y_,marker="*",label='BCast(49% Malicious)')

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
plt.plot(x_, y_,marker="s",label="Random")
#####

import matplotlib
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
xmajorLocator   = MultipleLocator(0.1)
xminorLocator   = MultipleLocator(0.05)
plt.gca().xaxis.set_major_locator(xmajorLocator)
plt.gca().xaxis.set_minor_locator(xminorLocator)


plt.xlim([0, 1])
plt.ylabel("Latency(ms)", fontsize=22)
plt.xlabel("Percentile", fontsize=22)
#plt.title("Placement under 30% inflation")
#plt.legend(ncol=2, loc='upper left',fontsize=10,fancybox=True,)
plt.legend(bbox_to_anchor=(0., 1.005, 1., .10), loc=3,
           ncol=3, mode="expand", borderaxespad=0., fontsize=12, markerscale=1.5)
plt.tick_params(labelsize=16)

plt.savefig('/Users/zengly/Downloads/P2PNetwork/test/graphs/no_response_attack.pdf',bbox_inches = 'tight')
plt.show()

