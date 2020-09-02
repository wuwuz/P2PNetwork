CFLAGS = -pg -g -Wall -std=c++11 -mpopcnt -march=native

all: sim

sim: sim.cpp
	g++ $(CFLAGS) -o sim sim.cpp 

clean:
	rm -f sim
