CFLAGS = -pg -g -Wall -std=c++11 -mpopcnt -march=native

all: sim coordinate_test

sim: sim.cpp
	g++ $(CFLAGS) -o sim sim.cpp 

coordinate_test: coordinate_test.cpp coordinate.h
	g++ $(CFLAGS) -o coordinate_test coordinate.h coordinate_test.cpp 

clean:
	rm -f sim
	rm -f coordinate_test
