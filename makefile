CFLAGS = -pg -g -Wall -std=c++17 #-mpopcnt -march=native

all: sim coordinate_test

sim: coordinate.h sim.cpp 
	g++-7 $(CFLAGS) -o sim sim.cpp 

coordinate_test: coordinate_test.cpp coordinate.h
	g++-7 $(CFLAGS) -o coordinate_test coordinate.h coordinate_test.cpp 

clean:
	rm -f sim
	rm -f coordinate_test
