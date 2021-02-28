#!usr/bin/python
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.figure(figsize=(8,4))

with open("/Users/zengly/Downloads/P2PNetwork/test/outputfile/inner_cluster.csv", "r") as f:
    data = f.readlines()

x_=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
yy = []
for k in data:
	klist = k.split(",")
	print klist
	yy.append(klist)

#maps={"K=1":"s", "K=2":"o", "K=4":"v", "K=6":"*", "K=8":"+", "K=10":"x", "K=12":"2", "K=14":"",}
maps={"dcluster=1":"s", "dcluster=2":"o", "dcluster=3":"*", "dcluster=4":"+", "dcluster=5":"x", "dcluster=6":"2", "dcluster=7":"v",}
#maps={"f=8":"s", "f=16":"o", "f=32":"v", "f=64":"*", "f=128":"d",}
#maps={"Random":"*", "PerigeeUBC":"d", "BlockP2P":"v", "BCastCluster":"o", "BCast":"s",}

for li in yy:
	name = li[0]
	y_ = [float(t) for t in li[1:-1]]
	plt.plot(x_, y_,marker=maps[name],label=name)


import matplotlib
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
xmajorLocator   = MultipleLocator(0.1)
xminorLocator   = MultipleLocator(0.05)
plt.gca().xaxis.set_major_locator(xmajorLocator)
plt.gca().xaxis.set_minor_locator(xminorLocator)


plt.xlim([0, 1])
plt.ylabel("Latency(ms)", fontsize=22)
plt.xlabel("Percentile", fontsize=22)
plt.legend(ncol=2, loc='lower right',fontsize=12,fancybox=True,) #lower right
#plt.legend(bbox_to_anchor=(0., 1.005, 1., .10), loc=3, 
#           ncol=3, mode="expand", borderaxespad=0., fontsize=10, markerscale=1.8)
plt.tick_params(labelsize=16)

plt.savefig('/Users/zengly/Downloads/P2PNetwork/test/graphs/inner_cluster_2.pdf',bbox_inches = 'tight')
plt.show()

