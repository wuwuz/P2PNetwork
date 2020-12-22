Test

200 nodes, 100 rounds

1.test random/fix neighbors
output:
vivaldi_200_100_16_random_fix.pdf
fix-neighbor is better(slightly)

2.test neighbor number
output:
vivaldi_200_100_64_32_16.pdf
64 neighbors is better

3.test system parameter Ce(weighted moving average of local error)
output:
vivaldi_200_100_64_ce.pdf
Ce=0.25 has smaller initial error

4.test system parameter Cc(time-step calculation)
output:
vivaldi_200_100_64_cc.pdf
Cc=0.25 has faster convergence

5.test node number performance
output:
vivaldi_node_100_64.pdf
200/400/800/1600 nodes
more nodes, higher initial error, but same convergence 

6.use real latency dataset(https://github.com/uofa-rzhu3/NetLatency-Data)
output:
vivaldi_planet_100_64.pdf
converge to 10ms(median) and 20ms(mean)

---------
490 PlanetLab Node Latency Dataset, 100 round, 64 neighbor, Ce=Cc=0.25
7.test 32random+32close neighbors
output:
vivaldi_planet_100_64.pdf
32random+32close has higher initial error, slower convergence, but slightly lower final error

---------
Attack
8. random breakdown(rtt is large)
output:
