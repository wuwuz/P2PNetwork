CFLAGS = -g -Wall -std=c++11 -mpopcnt -march=native

all: sim

sim: coordinate.h sim.cpp 
	g++ $(CFLAGS) -o sim sim.cpp 

clean:
	rm -f sim
	rm -f coordinate_test
