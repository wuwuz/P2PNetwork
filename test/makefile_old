CFLAGS = -g -Wall -std=c++11 -mpopcnt -march=native

all: coordinate_test

coordinate_test: coordinate_test.cpp coordinate.h
	g++ $(CFLAGS) coordinate.h coordinate_test.cpp 

clean:
	rm -f coordinate_test
