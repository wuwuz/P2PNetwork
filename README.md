# Introduction

The simulation source code in our paper **MERCURY: Fast Transaction Broadcast in High
Performance Blockchain Systems(Accepted by [INFOCOM 2023](https://infocom2023.ieee-infocom.org/program/accepted-paper-list-main-conference))** is available here.

**MERCURY** shortens the transaction propagation delay using two techniques: a virtual
coordinate system and an early outburst strategy. 
Our simulation results show that **MERCURY** outperforms prior propagation schemes and decreases overall propagation latency by up to 44% compared with Random mechanism.
When implemented in [Conflux](https://confluxnetwork.org/), an open-source high-throughput blockchain system, **MERCURY** reduces transaction propagation latency by over 50% with less than 5% bandwidth overhead.
Please review our paper for more details.

# Installation

In linux environment:

```
make
./sim
```

Main files:

1. `sim.cpp`: the simulation code
2. `coordinate.h`: the implementation of vivaldi algorithm
3. `geolocation.txt`: geolocation input, used by sim.cpp

# Citation

IF YOU USE THIS DATA SET IN ANY PUBLISHED RESEARCH, PLEASE KINDLY CITE THE FOLLOWING PAPER:

Mingxun Zhou\*, Liyi Zeng\*, Yilin Han, Peilun Li, Fan Long, Dong Zhou, Ivan Beschastnikh, Ming Wu, "MERCURY: Fast Transaction Broadcast in High Performance Blockchain Systems", IEEE INFOCOM 2023-IEEE Conference on Computer Communications. IEEE, 2023.(\* These authors contributed equally)

# Contacts

Email: mingxunz@andrew.cmu.edu, zengly17@mails.tsinghua.edu.cn.
