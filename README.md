# P2PNetwork

In linux environment:

```
make
./sim
./coordinate_test
```



Main files:

1. `sim.cpp`: the simulation code
2. `proj.py`: the visualization code
3. `ip_geo.py`: reads ip from `ip_res.csv`, converts them to geolocation. 
4. `geolocation.txt`: geolocation input, used by sim.cpp
5. `tree_struct.txt`: tree structure created by the simulation experiment
6. `coordinate.h`: the implementation of vivaldi algorithm
7. `coordinate_test.cpp`: test code for the vivaldi algorithm