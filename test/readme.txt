Test

200 nodes, 100 rounds

1.test random/fix neighbors
output:
vivaldi_200_100_16_random_fix.pdf
fix-neighbor is slightly better

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
490 PlanetLab Node Latency Dataset, 100 round, 64 neighbor, Ce=Cc=0.25
output:
vivaldi_planet_100_64.pdf
converge to 10ms(median) and 20ms(mean)

7.test 32random+32close neighbors
output:
vivaldi_planet_100_64.pdf
32random+32close has higher initial error, slower convergence, but slightly lower final error

8.virtual node coordiante placement
output:
vivaldi_planet_100_64_placement.pdf

9. test rtt stability(static/95%-105%/80%-120%, 10%/20%/30% breakdown)
output:
vivaldi_planet_100_64_rtt.pdf
vivaldi_planet_100_64_rtt_breakdown.pdf
impact error and convergence speed


---------
Newton Model
10. test performance
output:
newton_planet_100_64.pdf(centroid_shift_threshold=25ms)
newton_planet_100_64_placement.pdf 
lower (median)error than vivaldi

11. test rtt stability(static/95%-105%/80%-120%, 10%/20%/30% breakdown)
output:
newton_planet_100_64_rtt.pdf
newton_planet_100_64_breakdown.pdf
still performs well

---------
Attack
12. disorder attack: random coordinate, low error=0.01, rtt delay[100...1000]
output:
disorder_planet_100_64.pdf
disorder_after_planet_100_64.pdf(after 20 rounds)
newton performs better

13. deflation attack(malicious_cor=(0,0))
output:
deflation_planet_100_64.pdf
deflation_after_planet_100_64.pdf(after 20 rounds)
newton performs well

14. inflation attack
output:
inflation_400_planet_100_64.pdf(malicious_cor=(400,400))
obvious higher error
inflation_800_planet_100_64.pdf(malicious_cor=(800,800))
newton can not converge with 10% malicious nodes(why?)
