#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <queue>
#include <cmath>
#include <string>
#include <random>
#include <memory>
#include "coordinate.h"
#include <iostream>
#include <fstream>
//https://github.com/uofa-rzhu3/NetLatency-Data 
//RTTs between 490 nodes in the PlanetLab  


int main() {

    int N = 490;
    double planetLab_latency[N][N];

    FILE* f = fopen("PlanetLabData_1.txt", "r");

    for (int i = 0; i < N; i++) {
    	for (int j = 0; j < N; j++) 
        	fscanf(f, "%lf", &planetLab_latency[i][j]);
    }

    for (int i = 0; i < N; i++)
    	printf("%.3f\n", planetLab_latency[0][i]);

    return 0;
}


    