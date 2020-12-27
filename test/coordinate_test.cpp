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

#include <fstream>

#define memcle(a) memset(a, 0, sizeof(a))

using namespace std;
const int N = 3000;
const int TEST_ROUND = 100;

//dimension
const int REAL_COORDINATE_D = 3;
const int D = 2;

void larger_test() {

    //double world_coord[N][2];
    //int n;
    //FILE* f = fopen("geolocation.txt", "r");

    //fscanf(f, "%d", &n);
    //for (int i = 0; i < n; i++) {
    //    fscanf(f, "%lf%lf", &world_coord[i][0], &world_coord[i][1]);
    //}

    //n = 1600;
    //EuclideanVector<2> real_coord[N];
    //for (int i = 0; i < n; i++) 
    //    for (int j = 0; j < 2; j++)
    //        real_coord[i].v[j] = world_coord[i][j];


    //real latency dataset 490 nodes from PlanetLab
    int n = 490;
    double planetLab_latency[n][n];

    FILE* f = fopen("PlanetLabData_1.txt", "r");

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) 
            fscanf(f, "%lf", &planetLab_latency[i][j]);
    }

    VivaldiModel<D> model[N];
    for (int i = 0; i < n; i++)
        model[i] = VivaldiModel<D>(i);


    vector<double> err_median;
    vector<double> err_mean;
    vector<double> black_acc_median;

    //attack 10%20%30% random coordinate
    vector<int> malicious_nodes;
    while(malicious_nodes.size() <= 0.3*n){
        int id_ = rand() % n;
        vector<int>::iterator iter = std::find(malicious_nodes.begin(), malicious_nodes.end(), id_);
        while (iter != malicious_nodes.end())
        {
            id_ = rand() % n;
            iter = std::find(malicious_nodes.begin(), malicious_nodes.end(), id_);
        
        }
        malicious_nodes.push_back(id_);
    }


    for (int i = 0; i < TEST_ROUND * n; i++) {
        int x = i % n; 

        vector<int> selected_neighbor;
        if (model[x].have_enough_peer) {
            for (auto &y: model[x].random_peer_set)
                selected_neighbor.push_back(y);
        } else {
            //random 32 neighbors
            //for (int j = 0; j < 32; j++) {
            //    int y = rand() % n;
            //    while (y == x) y = rand() % n;
            //    selected_neighbor.push_back(y);
            //}
            //close 32 neighbors
            //for (int j = 0; j < 32; j++) {
            //    int y = rand() % n;
            //    while (planetLab_latency[y][x] > 100 or y == x) y = rand() % n;
            //    selected_neighbor.push_back(y);
            //}
            for (int j = 0; j < 64; j++) {
                int y = rand() % n;
                while (y == x) y = rand() % n;
                selected_neighbor.push_back(y);
            }
        }

        for (auto y: selected_neighbor)
        {
            double rtt = planetLab_latency[x][y]; //distance(real_coord[x], real_coord[y]) + 100;

            //stability rtt = 95%-105%, 80%-120%, 50%-150%
            //rtt = rtt * (rand()%11 + 95)/100.0;
            //rtt = rtt * (rand()%41 + 80)/100.0;
            //rtt = rtt * (rand()%101 + 50)/100.0;

            //breakdown rtt is large(1000ms) 10% 20% 30%
            //if (rand() % 100 < 30)
            //    rtt = 1000;

            Coordinate<D> cy = model[y].coordinate();

            //disorder attack: random coordinate, low error=0.01, rtt delay[100...1000]
            //vector<int>::iterator iter = std::find(malicious_nodes.begin(), malicious_nodes.end(), y);
            //if (iter != malicious_nodes.end() ){ //and i > 20 * n
            //    rtt = planetLab_latency[x][y] + (rand()%901+100);
            //    EuclideanVector<D> yy;
            //    yy.v[0] = rand()%601 + (-400);
            //    yy.v[1] = rand()%401 + (-200);
            //    cy = Coordinate<D>(yy, 0, 0.01);
            //}

            //deflation attack: coordinate(0,0), low error=0.01, 
            //vector<int>::iterator iter = std::find(malicious_nodes.begin(), malicious_nodes.end(), y);
            //if (iter != malicious_nodes.end() and i > 20 * n){ //
            //   rtt = planetLab_latency[x][y];
            //    EuclideanVector<D> yy;
            //    yy.v[0] = 0.0;
            //    yy.v[1] = 0.0;
            //    cy = Coordinate<D>(yy, 0, 0.01);
            //}

            //inflation attack: large coordinate, low error=0.01, 
            vector<int>::iterator iter = std::find(malicious_nodes.begin(), malicious_nodes.end(), y);
            if (iter != malicious_nodes.end()){ //and i > 20 * n
                rtt = planetLab_latency[x][y];
                EuclideanVector<D> yy;
                yy.v[0] = 800;
                yy.v[1] = 800;
                cy = Coordinate<D>(yy, 0, 0.01);
            }

            //double est_rtt = estimate_rtt(model[x].coordinate(), model[y].coordinate());
           // double relative_err = std::fabs(est_rtt - rtt) / rtt;

            //Coordinate<D> cx = model[x].coordinate();
            //Coordinate<D> cy = model[y].coordinate();

            //if (y == 0 && i > TEST_ROUND * 2 / 3)
            //    rtt = 10000;
            //if (rand() % 3 == 0) {
            //    rtt = 10000;
            //}

            model[x].observe(y, cy, rtt);


        }
        //every round 
        if (i != 0 && i%n == 0) {
            
            //EuclideanVector<D> centroid;

            //centroid = centroid + model[x].vector();
            //for (int y = 0; y < n; y++) {
            //    model[y].vector().show();
            //    printf("\n");
            //    centroid = centroid + model[y].vector();
            //}
            //centroid = centroid / (1.0 * n);

            //printf("centroid = ");
            //centroid.show();
            //printf("\n");
            //printf("centroid = %.2f\n", centroid.magnitude());

            //EuclideanVector<D> local_centroid;
            //local_centroid = model[x].vector();

            //for (auto y: selected_neighbor) 
            //    local_centroid = local_centroid + model[y].vector();

            //local_centroid = local_centroid / (1.0 * selected_neighbor.size() + 1);
            
            //printf("%d local centroid = ", x);
            
            //local_centroid.show();
            //printf("magnitude = %.2f\n", local_centroid.magnitude());
            

            vector<double> err_stat;
            for (int w = 0; w < n; w++)
                for (int z = w + 1; z < n; z++) {
                    double est_rtt = estimate_rtt(model[w].coordinate(), model[z].coordinate());
                    double real_rtt = planetLab_latency[w][z]; //distance(real_coord[i], real_coord[j]) + 100;
                    //printf("est = %.2f, real = %.2f\n", est_rtt, real_rtt);
                    if (real_rtt != 0) {
                        double abs_err = fabs(est_rtt - real_rtt); /// real_rtt;
                        err_stat.push_back(abs_err);
                    }
                }

            sort(err_stat.begin(), err_stat.end());

            printf("round %d", i/n);
            printf("err 50% = %.2f\n", err_stat[err_stat.size() / 2]);
            err_median.push_back(err_stat[err_stat.size() / 2]);
                
            //black list accuracy rate
            int num  = 0;
            vector<double> black_acc;
            vector<int>::iterator iter;
            std::unordered_set<int> bl;
            for (int w = 0; w < n; w++){
                bl = model[w].black_();
                num = 0;
                for(int z : bl){
                    iter = std::find(malicious_nodes.begin(), malicious_nodes.end(), z);
                    if(iter != malicious_nodes.end())
                        num += 1;
                }
                if (bl.size() == 0){
                    black_acc.push_back(-1);
                }
                else{
                    black_acc.push_back(num*1.0/bl.size());
                }
                
            }
            sort(black_acc.begin(), black_acc.end());

            printf("round %d", i/n);
            printf("black_acc 50% = %.2f\n", black_acc[black_acc.size() / 2]);
            black_acc_median.push_back(black_acc[black_acc.size() / 2]);

            //double mean_ = 0.0;
            //for(int i =0; i < err_stat.size(); i++)
            //    mean_ += err_stat[i];
            //printf("err mean = %.2f\n", mean_/err_stat.size());
            //err_mean.push_back(mean_/err_stat.size());
            //printf("err 90% = %.2f\n", err_stat[int(err_stat.size() * 0.9)]);
            //printf("err max = %.2f\n", err_stat[err_stat.size() - 1]);

        }

    }

    ofstream outf1; 

    //ofstream outf2;

    outf1.open("/Users/zengly/Downloads/P2PNetwork/test/outputs/err_median_inflation_30_800.txt");

    //outf2.open("/outputs/err_mean.txt");

    for (int i =0; i < err_median.size(); i++)
        outf1<<err_median[i]<<endl;

    //for (int i =0; i < err_mean.size(); i++)
    //    outf2<<err_mean[i]<<endl;

    outf1.close();
    //outf2.close();

    //coordinate 
    //ofstream outf; 
    //outf.open("coordinate.txt");
    //for (int i = 0; i < n; i++)
    //    outf<<model[i].coordinate().vector().v[0]<<","<<model[i].coordinate().vector().v[1]<<","<<model[i].coordinate().height()<<endl;
    //outf.close();

    //ofstream outf2; 

    //ofstream outf2;

    //outf2.open("/Users/zengly/Downloads/P2PNetwork/test/outputs/black_acc_median_inflation_30_800.txt");

    //outf2.open("/outputs/err_mean.txt");

    //for (int i =0; i < black_acc_median.size(); i++)
    //    outf2<<black_acc_median[i]<<endl;

    //for (int i =0; i < err_mean.size(); i++)
    //    outf2<<err_mean[i]<<endl;

    //outf2.close();

    
}


int main() {

    larger_test();

    return 0;
}