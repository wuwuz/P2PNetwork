#!usr/bin/python
# -*- coding: utf-8 -*-

import math

with open("/Users/zengly/Downloads/P2PNetwork/test/geolocation.txt", "r") as f:
    data = f.readlines()

print len(data)

xx = []
yy = []
for k in data[1:]:
	li_ = k.split(" ")
	xx.append(float(li_[0]))
	yy.append(float(li_[1]))
print xx[:5]
print yy[:5]

print max(xx)
print min(xx)

print max(yy)
print min(yy)