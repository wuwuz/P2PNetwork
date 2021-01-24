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
const int N = 3000;
const int TEST_ROUND = 100;
const double FIXED_DELAY = 200;

//dimension
const int REAL_COORDINATE_D = 3;
const int D = 2;

// 3 nodes test
void simple_3_nodes_test() {
    VivaldiModel<D> model[3];
    for (int i = 0; i < 3; i++)
        model[i] = VivaldiModel<D>(i);

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
                model[x].observe(y, model[y].coordinate(), delay[x][y]);
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

template<int D, typename T>
void calc_err_stat(VivaldiModel<D>* model, T* real_coord, int n, int mal_n = 0) {
    vector<double> err_stat;
    for (int i = 0; i < n - mal_n; i++)
        for (int j = i + 1; j < n - mal_n; j++) {
            double est_rtt = estimate_rtt(model[i].coordinate(), model[j].coordinate());
            double real_rtt = distance(real_coord[i], real_coord[j]) + 100;
            //printf("est = %.2f, real = %.2f\n", est_rtt, real_rtt);
            if (real_rtt != 0) {
                double abs_err = fabs(est_rtt - real_rtt) / real_rtt;
                err_stat.push_back(abs_err);
            }
        }

    sort(err_stat.begin(), err_stat.end());

    printf("err min = %.2f\n", err_stat[0]);
    for (int i = 5; i <= 9; i++)
        printf("err P%d %.2f\n", i * 10, err_stat[int(err_stat.size() * i / 10)]);
    printf("err max = %.2f\n", err_stat[err_stat.size() - 1]);

}

void larger_test() {

    double world_coord[N][2];
    int n;
    int mal_n;
    FILE* f = fopen("geolocation.txt", "r");

    fscanf(f, "%d", &n);
    for (int i = 0; i < n; i++) {
        fscanf(f, "%lf%lf", &world_coord[i][0], &world_coord[i][1]);
    }

    n = 1000;
    mal_n = n * 0.40;
    printf("malicious node = %d", mal_n);
    EuclideanVector<2> real_coord[N];
    for (int i = 0; i < n; i++) 
        for (int j = 0; j < 2; j++)
            real_coord[i].v[j] = world_coord[i][j];

    /*
    //EuclideanVector<REAL_COORDINATE_D> real_coord[N];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < REAL_COORDINATE_D; j++)
            real_coord[i].v[j] = random_between_0_1() * 500;
    }
    */

    VivaldiModel<D> model[N];
    for (int i = 0; i < n; i++)
        model[i] = VivaldiModel<D>(i, false, false, true);

    for (int test_round = 0; test_round < TEST_ROUND; test_round++) {
        if (test_round % 20 == 0) {
            printf("test round = %d\n", test_round);
            calc_err_stat(model, real_coord, n, mal_n);
        }
        for (int x = 0; x < n; x++) {
            //printf("%d\n", i);
            vector<int> selected_neighbor;
            if (model[x].have_enough_peer) {
                for (auto &y: model[x].random_peer_set)
                    selected_neighbor.push_back(y);

            } else {
                for (int j = 0; j < 16; j++) {
                    int y = rand() % n;
                    while (y == x) y = rand() % n;
                    selected_neighbor.push_back(y);
                }
            }

            for (auto y: selected_neighbor)
            {
                double rtt = distance(real_coord[x], real_coord[y]) + 100;
                //double est_rtt = estimate_rtt(model[x].coordinate(), model[y].coordinate());
                //double relative_err = std::fabs(est_rtt - rtt) / rtt;

                //Coordinate<D> cx = model[x].coordinate();
                Coordinate<D> cy = model[y].coordinate();

                //if (y == 0 && i > TEST_ROUND * 2 / 3)
                //    rtt = 10000;
                if (y < mal_n && test_round > TEST_ROUND / 3) {
                    double tmp[2] = {random_between_0_1() * 100, random_between_0_1() * 100};
                    //double tmp[2] = {500, 500};
                    cy = Coordinate<D>(EuclideanVector<D>(tmp), 100, 0.1);
                }

                //if (rand() % 3 == 0) {
                //    rtt = 2000;
               // }

                model[x].observe(y, cy, rtt);
                //model[y].observe(x, cx, rtt);

                //double new_est_rtt = estimate_rtt(model[x].coordinate(), model[y].coordinate());
                //double new_err = std::fabs(new_est_rtt - rtt) / rtt;

                
                /*
                if (new_err > relative_err * 1.2) {
                    printf("Increasing error: i = %d, x = %d, y = %d\n", i, x, y);
                    printf("rtt = %.2f, old est = %.2f, new est = %.2f\n", rtt, est_rtt, new_est_rtt);
                    printf("old = %.2f, new = %.2f\n", relative_err, new_err);
                }
                */
            }

            if (x == 10) {
                EuclideanVector<D> centroid;
                //centroid = centroid + model[x].vector();
                for (int y = 0; y < n; y++) {
                    //model[y].vector().show();
                    //printf("\n");
                    centroid = centroid + model[y].vector();
                }
                centroid = centroid / (1.0 * n);

                printf("centroid = ");
                centroid.show();
                printf("centroid = %.2f\n", centroid.magnitude());

                EuclideanVector<D> local_centroid;
                local_centroid = model[x].vector();
                for (auto y: selected_neighbor) 
                    local_centroid = local_centroid + model[y].vector();
                local_centroid = local_centroid / (1.0 * selected_neighbor.size() + 1);
                printf("%d's local centroid = ", x);
                local_centroid.show();
                printf("magnitude = %.2f\n", local_centroid.magnitude());

            }
            //printf("%d ", x);
            //model[x].coordinate().show();
            //printf("\n");
        }
    }

    calc_err_stat(model, real_coord, n, mal_n);
}


int main() {
    //simple_3_nodes_test();
    larger_test();
    return 0;
}