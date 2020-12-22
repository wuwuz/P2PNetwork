#!usr/bin/python
# -*- coding: utf-8 -*-

import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter



with open("/Users/zengly/Downloads/P2PNetwork/test/PlanetLabData_1.txt", "r") as f:
    data = f.readlines()

print len(data)

ma={}
i = 0
for line in data:
	li = line.split('	')
	j = 0
	for k in li:
		if i not in ma:
			ma[i] = {}
		if j not in ma[i]:
			ma[i][j] = 0
		ma[i][j] = float(k.replace("\n",""))
		j+=1
	i+=1

print len(ma)
for i in range(0, 490):
	if ma[i][i] != 0:
		print "no"
