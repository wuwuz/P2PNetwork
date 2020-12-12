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
#define memcle(a) memset(a, 0, sizeof(a))

using namespace std;

//dimension
const int REAL_COORDINATE_D = 3;
const int D = 3;

// 3 nodes test
void simple_3_nodes_test() {
    VivaldiModel<D> model[3];
    double delay[3][3] = {
        {0, 300, 400},
        {300, 0, 500},
        {400, 500, 0}
    };

    // the delay between 1 and 2 --- 300 ms
    // the delay between 1 and 3 --- 400 ms
    // the delay between 2 and 3 --- 500 ms
    
    for (int i = 0; i < 100; i++) {
        int x = i % 3;
        for (int y = 0; y < 3; y++)
            if (x != y) {
                model[x].observe(model[y].coordinate(), delay[x][y]);
            }
    }

    for (int i = 0; i < 3; i++) {
        model[i].coordinate().show();
        printf("\n");
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i != j) {
                printf("%.2f ", 
                    estimate_rtt(model[i].coordinate(), model[j].coordinate())
                );
            } else {
                printf("%.2f ", 0.0);
            }
        }
        printf("\n");
    }
}
void larger_test() {
    const int n = 1000;

    EuclideanVector<REAL_COORDINATE_D> real_coord[n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < REAL_COORDINATE_D; j++)
            real_coord[i].v[j] = random_between_0_1() * 500;
    }

    VivaldiModel<D> model[n];

    for (int i = 0; i < 30 * n; i++) {
        int x = i % n; 
        //printf("%d\n", i);
        for (int j = 0; j < std::min(16, n); j++) 
        {
            int y = rand() % n;
            while (y == x) y = rand() % n;

            double rtt = distance(real_coord[x], real_coord[y]);
            double est_rtt = estimate_rtt(model[x].coordinate(), model[y].coordinate());
            double relative_err = std::fabs(est_rtt - rtt) / rtt;

            model[x].observe(model[y].coordinate(), distance(real_coord[x], real_coord[y]));
            double new_est_rtt = estimate_rtt(model[x].coordinate(), model[y].coordinate());
            double new_err = std::fabs(new_est_rtt - rtt) / rtt;

            if (new_err > relative_err) {
                printf("Increasing error: i = %d, x = %d, y = %d\n", i, x, y);
                printf("rtt = %.2f, old est = %.2f, new est = %.2f\n", rtt, est_rtt, new_est_rtt);
                printf("old = %.2f, new = %.2f\n", relative_err, new_err);
            }
        }
        //printf("%d ", x);
        //model[x].coordinate().show();
        //printf("\n");
    }

    vector<double> err_stat;
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            double est_rtt = estimate_rtt(model[i].coordinate(), model[j].coordinate());
            double real_rtt = distance(real_coord[i], real_coord[j]);
            //printf("est = %.2f, real = %.2f\n", est_rtt, real_rtt);
            if (real_rtt != 0) {
                double abs_err = fabs(est_rtt - real_rtt) / real_rtt;
                err_stat.push_back(abs_err);
            }
        }

    sort(err_stat.begin(), err_stat.end());

    printf("err min = %.2f\n", err_stat[0]);
    printf("err max = %.2f\n", err_stat[err_stat.size() - 1]);
    printf("err mean = %.2f\n", err_stat[err_stat.size() / 2]);
}


int main() {
    //simple_3_nodes_test();
    larger_test();
    return 0;
}