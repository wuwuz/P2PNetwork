#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <queue>
#include <cmath>
#include <string>
#include <random>
#define memcle(a) memset(a, 0, sizeof(a))

using namespace std;
const int N = 8500;
const double pi = acos(-1);
const double R = 6371000; // radius of the earth
const double inf = 1e8;
const int MAX_DEPTH = 20;
//typedef unsigned int int;
int n;
mt19937 rd(1000);
bool recv_flag[N];
int recv_parent[N];
double recv_time[N]; 
int depth[N];

int mal_flag[N];


// coordinate, using longitude and latitude
class coordinate {
  public:
    double lat, lon;
};

coordinate coord[N];
// sorted_list[u] : index list sorted by the distance between nodes and the origin u
int sorted_list[N][N];

// from degree to radian 
double rad(double deg) {return deg * pi / 180;}

// distance between two coordinate
double distance(const coordinate &a, const coordinate &b) {
    if (abs(a.lat - b.lat) < 0.1 && abs(a.lon - b.lon) < 0.1)
        return 0;
    double latA = rad(a.lat), lonA = rad(a.lon);
    double latB = rad(b.lat), lonB = rad(b.lon);
    double C = cos(latA) * cos(latB) * cos(lonA - lonB) + sin(latA) * sin(latB);
    double dist = acos(C) * R ;
    return dist / 100000;
}

class message {
  public:
    int root, src, dst, step;
    double send_time, recv_time;

    message(int _root, int _src, int _dst, int _step, double _send_time, double _recv_time) : 
        root(_root), src(_src), dst(_dst), step(_step), send_time(_send_time), recv_time(_recv_time) {

    }

    void print_info() {
        fprintf(stderr, "message rooted at %d sent from node %d to %d\n, step %d", root, src, dst, step);
        fprintf(stderr, "send time at %.2f, recv time at %.2f, delay is %.2f\n", send_time, recv_time, recv_time - send_time);
    }
};

bool operator>(const message &a, const message &b) {
    return a.recv_time > b.recv_time;
}

class graph {
  private: 
    vector<int> in_bound[N];
    vector<int> out_bound[N];
    int n; 
    int m;

  public:
    graph(int _n) : n(_n) {
       m = 0; 
    }

    bool add_edge(int u, int v) {
        // avoid self-loop and duplicate edge
        if (u == v) return false;
        for (auto nb_u : out_bound[u]) 
            if (nb_u == v) 
                return false;
        out_bound[u].push_back(v);
        in_bound[v].push_back(u);
        m++;
        return true;
    }

    vector<int> outbound(int u) {
        auto v = out_bound[u];
        return v;
    }
    vector<int> inbound(int u) {
        auto v = in_bound[u];
        return v;
    }

    void print_info() {
        double avg_outbound = 0;
        for (int i = 0; i < n; i++) {
            fprintf(stderr, "node %d's outbound:", i);
            avg_outbound += out_bound[i].size();
            for (auto j : out_bound[i])
                fprintf(stderr, " %d", j);
            fprintf(stderr, "\n");
        }
        avg_outbound /= n;
        fprintf(stderr, "%.2f\n", avg_outbound);
    }
};

int random_num(int n) {
    return rand() % n;
}

class basic_algo {
// strategy base class
//
// respond(msg): 
// one message delivered at [msg.dst],
// return the index list of its choice of relay nodes
//

  public:
    basic_algo() {}
    virtual vector<int> respond(message msg) = 0;
    //static const char* get_algo_name();
};

class random_flood : public basic_algo {

// random flood : 
// 1. Connet the graph as a ring to prevent partition
// 2. Every node selects other 7 random outbounds

  private: 
    graph G; // random graph
    const int deg = 8;
    static constexpr const char* algo_name = "random_flood8";

  public:
    random_flood(int n, coordinate *coord, int root = 0) : G(n) {
        // firstly connect a ring, then random connect

        for (int u = 0; u < n; u++)
            G.add_edge(u, (u + 1) % n);

        for (int u = 0; u < n; u++) {
            int dg = deg - 1;
            //if (u == root)
            //    dg = 32 - 1;
            for (int k = 0; k < dg; k++) {
                int v = random_num(n);
                while (G.add_edge(u, v) == false)
                    v = random_num(n);
            }
        }
    }

    vector<int> respond(message msg)  {
        // Directly return all [msg.dst]'s outbounds.

        int u = msg.dst;
        vector<int> nb_u = G.outbound(u);
        vector<int> ret;
        for (auto v : nb_u) 
            if (v != msg.src) 
                ret.push_back(v);
        return ret;
    }

    static const char* get_algo_name() {return algo_name;} 
    void print_info() {
        G.print_info();
    }
};

// the difference of lontitudes should be in the range of [-180, 180]
double fit_in_a_ring(double x)  {
    if (x < -180) x += 360;
    if (x > 180) x -= 360;
    return x;
}

// the angle between the vector r ---> u and u ---> v should be in [-90, 90]
// notice: simply use (lontitude, latitude) as the normal 2-d (x, y) coordinate
bool angle_check(const coordinate &r, const coordinate &u, const coordinate &v) {
    double x1 = u.lon - r.lon, y1 = u.lat - r.lat;
    double x2 = v.lon - u.lon, y2 = v.lat - u.lat;
    x1 = fit_in_a_ring(x1);
    x2 = fit_in_a_ring(x2);

    // get the vertical vector of (u - r)
    double x3 = y1, y3 = -x1;

    // use cross dot to check the angle
    return (x3 * y2 - x2 * y3) > -1e-3;
}

template <bool angle_check_flag = false>
class from_near_to_far : public basic_algo {
// from_near_to_far:
// Suppose the broadcast root is [root].
// Firstly, sort the nodes based on the distance between them and [root].
//
// When a node [u] receives a message, it choose 3 kinds of outbounds:
// 1. Look at the nearest nodes of [u]. If their distances to [root] are larger than [u], 
//    they can be selected as [u]'s outbounds. (at most 4 outbounds)
// 2. Randomly select 2 nodes that have larger distances to [root] than [u].
// 3. Randomly select 2 nodes.


// The (4, 2, 2) combination seems good. Tried (3, 3, 2), (5, 1, 2)... 
// If angle_check_flag == true, the type 1 and 2 outbounds should also satisfy the angle_check condition

  private: 
    graph G; // random graph
    const int random_out = 4;
    const int dist_out = 4;
    static constexpr const char* algo_name = "near_to_far";

  public:
    vector<pair<double, int> > dist_list;
    vector<int> rank;

    from_near_to_far(int n, coordinate *coord, int root = 0) : G(n), rank(n, 0) {
        // get the index of the sorted list origin at [root]
        rank[root] = 0;
        for (int i = 0; i < n - 1; i++)
            rank[sorted_list[root][i]] = i + 1;

        // every node has 4 random outbound (but only 2 will be used in one response)
        for (int u = 0; u < n; u++) {
            for (int k = 0; k < 4; k++) {
                int v = random_num(n);
                while (G.add_edge(u, v) == false)
                    v = random_num(n);
            }
        }
    }

    vector<int> respond(message msg)  {
        const int near_outbound = 4;
        const int random_outbound = 2;
        const int totally_random_outbound = 2;
        vector<int> ret;
        int cnt = 0;
        int u = msg.dst;

        // 2 totally random outbound
        //uniform_int_distribution<> range(0, n - 1);
        cnt = 0;
        for (; cnt < totally_random_outbound; ) {
            int v = random_num(n);
            if (v != u && v != msg.src && find(ret.begin(), ret.end(), v) == ret.end()) {
                ret.push_back(v);
                cnt++;
            }
        }
    
        // 2 random outbound, ensure distance
        cnt = 0;
        int trial = 0;
        //uniform_int_distribution<> range(i, n - 2);
        for (; cnt < random_outbound && trial < 100; trial++) {
            //int v = sorted_list[u][range(rd)];
            int v = random_num(n);
            if (v != u && v != msg.src && rank[v] > rank[u] && find(ret.begin(), ret.end(), v) == ret.end()) {
                if (angle_check_flag == true && angle_check(coord[msg.root], coord[u], coord[v]) == false) continue;
                ret.push_back(v);
                cnt++;
            }
        }

        // near outbound, but ensure they are further than the current node
        for (int i = 0; i < n - 1 && cnt < near_outbound; i++) {
            int v = sorted_list[u][i];
            if (v != u && v != msg.src && rank[v] > rank[u] && find(ret.begin(), ret.end(), v) == ret.end()) {
                if (angle_check_flag == true && angle_check(coord[msg.root], coord[u], coord[v]) == false) continue;
                ret.push_back(v);
                cnt++;
            }
        }
        reverse(ret.begin(), ret.end());

        return ret;
    }
    static const char* get_algo_name() {return algo_name;} 
    void print_info() {
        G.print_info();
    }
};

class static_build_tree : public basic_algo {
// static_build_tree:
// Suppose the broadcast root is [root].
// Firstly, sort the nodes based on the distance between them and [root].
// The sorted list is [list].
//
// Build the broadcast tree as following rules:
// For the node, u = list[i]
// The father should be:
//    v in list[0...i-1]:
//    minimize: tree_distance(root, v) + distance(v, u)
//    subject to: out_bound[v] < 8 

  private: 
    graph G; // random graph
    const int random_out = 4;
    const int dist_out = 8;
    static constexpr const char* algo_name = "static_build";
    double dist[N];
    int out_bound_cnt[N];
    int list[N];
  
  public: 
    static_build_tree(int n, coordinate *coord, int root = 0) : G(n) {
        memcle(dist);
        memcle(out_bound_cnt);
        memcle(list);

        list[0] = root;
        for (int i = 0; i < n - 1; i++) {
            int u = sorted_list[root][i];
            list[i + 1] = u;

            double cur_min = 1e100;
            int cur_parent = 0;
            for (int j = 0; j <= i; j++) {
                int v = list[j];
                if ((v == root && out_bound_cnt[v] < 32) || (out_bound_cnt[v] < dist_out && dist[v] + distance(coord[u], coord[v]) + 50 < cur_min)) {
                    cur_min = distance(coord[u], coord[v]) + dist[v] + 50;
                    cur_parent = v;
                }
            }
            // set parent of u 
            G.add_edge(cur_parent, u);
            dist[u] = cur_min;
            out_bound_cnt[cur_parent]++;
        }
    }

    vector<int> respond(message msg)  {
        int u = msg.dst;
        vector<int> nb_u = G.outbound(u);
        vector<int> ret;
        for (auto v : nb_u) 
            if (v != msg.src) 
                ret.push_back(v);
        return ret;
    }

    static const char* get_algo_name() {return algo_name;} 
    void print_info() {
        G.print_info();
    }
};

const static int K = 8;
const static int max_iter = 100;
coordinate center[K];
coordinate avg[K];
int cluster_cnt[K];
int cluster_result[N];
vector<int> cluster_list[K];

// k_means: 8 cluster
// cluster_cnt[i]: #nodes in cluster i
// cluster_list[i]: list of nodes in cluster i
// cluster_result[u]: u belongs to this cluster

void k_means() {
    vector<int> tmp_list;

    for (int i = 0; i < K; i++) {
        while (true) {
            int u = random_num(n);
            if (find(tmp_list.begin(), tmp_list.end(), u) == tmp_list.end()) {
                center[i] = coord[u];
                tmp_list.push_back(u);
                break;
            }
        }
    }

    // K means
    for (int iter = 0; iter < max_iter; iter++) {

        // find the nearest center
        for (int i = 0; i < n; i++) {
            double dist = 1e100;
            int cur_cluster = 0;
            for (int j = 0; j < K; j++)
                if (distance(center[j], coord[i]) < dist) {
                    dist = distance(center[j], coord[i]);
                    cur_cluster = j;
                }
            cluster_result[i] = cur_cluster;
        }

        // re-calculate center
        memcle(avg);
        memcle(cluster_cnt);
        for (int i = 0; i < n; i++) {
            avg[cluster_result[i]].lon += coord[i].lon;
            avg[cluster_result[i]].lat += coord[i].lat;
            cluster_cnt[cluster_result[i]]++;
        }
        for (int i = 0; i < K; i++) 
            if (cluster_cnt[i] > 0) {
                center[i].lon = avg[i].lon / cluster_cnt[i];
                center[i].lat = avg[i].lat / cluster_cnt[i];
            }
    }

    //for (int i = 0; i < K; i++)
    //    fprintf(stderr, "%d ", cluster_cnt[i]);

    for (int i = 0; i < n; i++) 
        cluster_list[cluster_result[i]].push_back(i);
}

template <int root_deg_per_cluster = 1>
class k_means_cluster : public basic_algo {
// k_means_cluster:
// firstly build K clusters (K = 8)
// For the [root], it randomly connects to root_deg_per_cluster nodes in every cluster. (1, 2, 4...)
// For other nodes, they randomly connects to 6 nodes in the same cluster and 2 nodes in other clusters.

  private: 
    graph G; // random graph
    const int random_out = 4;
    const int dist_out = 4;
    static constexpr const char* algo_name = "cluster";


  public: 
    k_means_cluster(int n, coordinate *coord, int root = 0) : G(n) {
        
        // for every cluster, root builds root_deg_per_cluster connections
        for (int i = 0; i < K; i++) 
            if (cluster_cnt[i] > 0) {
                for (int trial = 0, cnt = 0; trial < 100 && cnt < root_deg_per_cluster; trial++) {
                    int u = cluster_list[i][random_num(cluster_list[i].size())];
                    if (u != root && G.add_edge(root, u))
                        cnt++;
                }
            }

        //other nodes
        for (int i = 0; i < n; i++) 
            if (i != root) {
                int c = cluster_result[i];
                // 6 out_bound in the same cluster
                if (cluster_cnt[c] <= 7) {
                    for (int j : cluster_list[c])
                        if (i != j)
                            G.add_edge(i, j);
                } else {
                    for (int trial = 0, cnt = 0; trial < 100 && cnt < 6; trial++) {
                        int j = cluster_list[c][random_num(cluster_cnt[c])];
                        if (i != j && G.add_edge(i, j))
                            cnt++;
                    }
                }

                // 2 random out bounds to other cluster
                for (int trial = 0, cnt = 0; trial < 100 && cnt < 2; trial++) {
                    int c_other = random_num(K);
                    if (c_other != c && cluster_cnt[c_other] > 0) {
                        int j = cluster_list[c_other][random_num(cluster_cnt[c_other])];
                        if (G.add_edge(i, j))
                            cnt++;
                    }
                }

                // 6 more out_bound for every depth-2 node
                for (int trial = 0, cnt = 0; trial < 100 && cnt < 6; trial++) {
                    int j = cluster_list[c][random_num(cluster_cnt[c])];
                    if (i != j && G.add_edge(i, j))
                        cnt++;
                }

                // 2 more random out bounds to other cluster
                for (int trial = 0, cnt = 0; trial < 100 && cnt < 2; trial++) {
                    int c_other = random_num(K);
                    if (c_other != c && cluster_cnt[c_other] > 0) {
                        int j = cluster_list[c_other][random_num(cluster_cnt[c_other])];
                        if (G.add_edge(i, j))
                            cnt++;
                    }
                }
            }
    }
        
    vector<int> respond(message msg)  {
        int u = msg.dst;
        vector<int> nb_u = G.outbound(u);
        vector<int> ret;
        int cnt = 0;
        for (auto v : nb_u) 
            if (v != msg.src) {
                ret.push_back(v);
                cnt++;
                if (msg.step > 1 && cnt >= 8) break;
                //if (msg.step > 3 && cnt >= 2) break;
            }
        return ret;
    }

    static const char* get_algo_name() {return algo_name;} 
    void print_info() {
        G.print_info();
    }
};

class block_p2p : public basic_algo {
// block p2p
// firstly build K clusters (K = 8)
// Inside a cluster, it connects Chord-type graph
// Every cluster has one entry point. One entry point connects to all other entry points.

  private: 
    graph G; // random graph
    const int random_out = 4;
    const int dist_out = 4;
    static constexpr const char* algo_name = "blockp2p";


  public: 
    block_p2p(int n, coordinate *coord, int root = 0) : G(n) {
        // the first node in every cluster's list is the entry points
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++)
                if (i != j) {
                    G.add_edge(cluster_list[i][0], cluster_list[j][0]);
                }
            
        // connect a Chord-type graph
        for (int i = 0; i < K; i++) {
            int cn = cluster_cnt[i];
            for (int j = 0; j < cn; j++) {
                int u = cluster_list[i][j];
                if (cn <= 8) {
                    // if the cluster size is small, connect it as a fully-connected graph
                    for (auto v : cluster_list[i])
                        if (u != v)
                            G.add_edge(u, v);
                } else {
                    // Chord-type graph
                    for (int k = 1; k < cn; k *= 2)  
                        G.add_edge(u, cluster_list[i][(j + k) % cn]); // connect u and (u + 2^k) mod cn
                    G.add_edge(u, cluster_list[i][(j + cn / 2) % cn]); // connect the diagonal
                }
                //G.add_edge(u, cluster_list[i][0]);
            }
        }
    }
        
    vector<int> respond(message msg)  {
        int u = msg.dst;
        vector<int> nb_u = G.outbound(u);
        vector<int> ret;
        //int cnt = 0;
        for (auto v : nb_u) 
            if (v != msg.src) {
                ret.push_back(v);
                //if (msg.step > 3 && cnt >= 2) break;
            }
        return ret;
    }

    static const char* get_algo_name() {return algo_name;} 
    void print_info() {
        G.print_info();
    }
};



class test_result {
  public : 
    double dup_rate;
    vector<double> latency;
    double depth_cdf[MAX_DEPTH];

    test_result() : dup_rate(0), latency(21, 0) {
        memcle(depth_cdf);
    }
    void print_info() {
        fprintf(stderr, "dup_rate");
        for (int i = 0; i < 21; i++)
            fprintf(stderr, ", %.2f", i * 0.05);
        fprintf(stderr, "\n");
        fprintf(stderr, "%.2f", dup_rate);
        for (int i = 0; i < 21; i++)
            fprintf(stderr, ", %.2f", latency[i]);
        fprintf(stderr, "\n");
    }
};

template <class algo_T>
test_result single_root_simulation(int root, int rept_time = 1, double mal_node = 0.0) {
// Test the latency of the message originated from [root].
// 1) Use a global heap to maintain the message queue and fetch the next delivered message.
// 2) For every delivered message, ignore it if it is a duplicated message.
// 3) Otherwise, invoke algo_T's respond function to get the relay node list.
// 4) Then create new messages to the relay nodes.

// Delay model: 
// When a node receives a message, it has 50ms delay to handle it. Then it sends to other nodes without delay.

    //srand(100);
    test_result result;

    // initialize test algorithm class
    algo_T algo(n, coord, root);
    //algo.print_info();

    for (int rept = 0; rept < rept_time; rept++)  {

        priority_queue<message, vector<message>, greater<message> > msg_queue;
        msg_queue.push(message(root, root, root, 0, 0, 0)); // initial message

        memcle(recv_flag);
        memcle(recv_time);
        memset(recv_parent, -1, sizeof(recv_parent));
        memcle(depth);
        vector<int> recv_list;

        int dup_msg = 0;

        for (; !msg_queue.empty(); ) {
            message msg = msg_queue.top();
            msg_queue.pop();

            int u = msg.dst; // current node



            // malicious node -- no response
            if (mal_flag[u] == true) continue;

            // duplicate msg -- ignore
            if (recv_flag[u] == true) {
                dup_msg++;
                continue;
            }
            //msg.print_info();

            recv_flag[u] = true;
            recv_time[u] = msg.recv_time;
            recv_parent[u] = msg.src;
            recv_list.push_back(u);
            if (u != root)
                depth[u] = depth[msg.src] + 1;

            auto relay_list = algo.respond(msg);
            double delay_time = 50; // delay_time = 10ms per link
            //double delay_time = 0; // delay_time = 10ms per link
            if (u == root) delay_time = 0;
            for (auto v : relay_list) {
                double dist = distance(coord[u], coord[v]) + 20; // rtt : 10 + distance(u, v)
                // TODO: Add random delay in transmission
                message new_msg = message(root, u, v, msg.step + 1, recv_time[u] + delay_time, recv_time[u] + dist + delay_time);
                msg_queue.push(new_msg);
            }
        }

        for (int i = 0; i < n; i++)
            if (recv_flag[i] == false && mal_flag[i] == false) {
                recv_time[i] = inf;
                recv_list.push_back(i);
                depth[i] = MAX_DEPTH - 1; // depth = 19 ---- uncovered node
            }

        int non_mal_node = recv_list.size();
        result.dup_rate += (double(dup_msg) / (dup_msg + non_mal_node));

        for (int u: recv_list)
            result.depth_cdf[depth[u]] += 1;
        for (int i = 0; i < MAX_DEPTH; i++)
            result.depth_cdf[i] /= non_mal_node;

        int cnt = 0;
        for (double pct = 0.05; pct <= 1; pct += 0.05, cnt++) {
            int i = non_mal_node * pct;
            result.latency[cnt] += recv_time[recv_list[i]];
            //if (pct == 0.95 && recv_time[recv_list[i]] < 400)
            //    fprintf(stderr, "strange root %d, node %d\n", root, recv_list[i]);
        }

    }


    result.dup_rate /= rept_time; 
    for (int i = 0; i < MAX_DEPTH; i++) 
        result.depth_cdf[i] /= rept_time; 
    for (size_t i = 0; i < result.latency.size(); i++) {
        double tmp = int(result.latency[i] / inf);
        result.latency[i] -= tmp * inf;
        if (rept_time - tmp == 0)
            result.latency[i] = 0;
        else
            result.latency[i] /= (rept_time - tmp);

        if (result.latency[i] < 0.1)
            result.latency[i] = inf;
    }


    // Print the tree structure (only when the root is 0)

    if (root == 0) {
        FILE* pf = fopen("tree_struct.txt", "w");
        if (pf != NULL) {
            fprintf(pf, "%d %d\n", n, root);
            for (int i = 0; i < n; i++) {
                fprintf(pf, "%d\n", recv_parent[i]);
            }
        } else 
            fprintf(stderr, "cannot open tree_struct.txt\n");
    }
              
    return result;
}

template <class algo_T>
test_result simulation(int rept_time = 1, double mal_node = 0.0) {

// Test the latency and duplication rate for the whole network.
// Firstly ranomly select some malicious nodes.
// Then, for every honest node, do a single_root_simulation.

    srand(100);

    test_result result;

    FILE* output = fopen("sim_output.csv", "a");
    if (output == NULL) {
        fprintf(stderr, "cannot open file\n");
        return result;
    }

    int test_time = 0;
    for (int rept = 0; rept < rept_time; rept++) {
        //fprintf(stderr, "rept %d\n", rept);
        // 1) generate malicious node list
        memcle(mal_flag);
        for (int i = 0; i < mal_node * n; i++) {
            int picked_num = random_num(n);
            while (mal_flag[picked_num] == true)  
                picked_num = random_num(n);
            mal_flag[picked_num] = true;
        }

        //for (int i = 0; i < K; i++)
        //    if (mal_flag[cluster_list[i][0]] == true)
        //        fprintf(stderr, "entry of cluster %d --- %d is malicious\n", i, cluster_list[i][0]);
        
        //for (int i = 0; i < n; i++)
        //    fprintf(stderr, "%d", mal_flag[i]);
        

        // 2) simulate the message at source i
        //int normal_node = n - mal_node * n;
        for (int root = 0; root < n; root++) {
            if (mal_flag[root] == false) {
                //fprintf(stderr, "%d", i);
                test_time++;
                auto res = single_root_simulation<algo_T>(root, 1, mal_node);
                result.dup_rate += res.dup_rate;
                for (size_t i = 0; i < result.latency.size(); i++) {
                    result.latency[i] += res.latency[i];
                    //if (i == 19 && result.latency[i] > 0)
                    //    fprintf(stderr, "root %d latency sum at %.2f %.2f\n", root, 0.05 * i, result.latency[i]);
                }
                for (int i = 0; i < MAX_DEPTH; i++)
                    result.depth_cdf[i] += res.depth_cdf[i];
            }
        }
    }

    result.dup_rate /= test_time;
    for (size_t i = 0; i < result.latency.size(); i++) {
        double tmp = int(result.latency[i] / inf);
        result.latency[i] -= tmp * inf;
        result.latency[i] /= (test_time - tmp);
        if (test_time - tmp == 0)
            result.latency[i] = 0;
    }
    for (int i = 0; i < MAX_DEPTH; i++)
        result.depth_cdf[i] /= test_time;

    //fprintf(stderr, "latency sum at 0.95 %.2f\n", result.latency[19]);
    fprintf(output, "%s\n", algo_T::get_algo_name());
    printf("%s\n", algo_T::get_algo_name());
    fprintf(output, "#node, mal node, Dup Rate, ");
    printf("#node, mal node, Dup Rate, ");
    for (double p = 0.05; p <= 1; p += 0.05) {
        fprintf(output, "%.2f, ", p);
        printf("%.2f, ", p);
    }
    fprintf(output, "\n");
    printf("\n");

    fprintf(output, "%d, %.2f, %.2f, ", n, mal_node, result.dup_rate);
    printf("%d, %.2f, %.2f, ", n, mal_node, result.dup_rate);
    //printf
    int cnt = 0;
    for (double p = 0.05; p <= 1; p += 0.05, cnt++) {
        fprintf(output, "%.2f, ", result.latency[cnt]);
        printf("%.2f, ", result.latency[cnt]);
    }
    fprintf(output, "\n");
    printf("\n");

    fprintf(output, "depth pdf\n");
    printf("depth pdf\n");
    for (int i = 0; i < MAX_DEPTH; i++) {
        fprintf(output, "%d, ", i);
        printf("%d, ", i);
    }
    fprintf(output, "\n");
    printf("\n");

    for (int i = 0; i < MAX_DEPTH; i++) {
        fprintf(output, "%.4f, ", result.depth_cdf[i]);
        printf("%.4f, ", result.depth_cdf[i]);
    }
    fprintf(output, "\n");
    printf("\n");

    fclose(output);
    return result;
}

void init() { 
    // Read the geo information from input.
    // For every node [u], sorted all the nodes based on the distance to [u], stored in sorted_list[u].

    n = 0;
    FILE* f = fopen("geolocation.txt", "r");
    fscanf(f, "%d", &n);
    for (int i = 0; i < n; i++) {
        fscanf(f, "%lf%lf", &coord[i].lat, &coord[i].lon);
    }

    n = 500;
    
    for (int i = 0; i < n; i++) {
        vector<pair<double, int> > rk;
        for (int j = 0; j < n; j++) {
            if (i != j) {
                rk.push_back(make_pair(distance(coord[i], coord[j]), j));
            } else {
            }
        }
        sort(rk.begin(), rk.end());
        for (int j = 0; j < n - 1; j++) {
            sorted_list[i][j] = rk[j].second;
        }
    }

    fclose(f);
}

int main() {
    int rept = 5;
    double mal_node = 0.0;
    init();
    k_means();

    for (int i = 0; i < 10; i++) {
        simulation<random_flood>(rept, mal_node);
        //simulation<from_near_to_far<false> >(rept, mal_node);
        simulation<from_near_to_far<true> >(rept, mal_node);
        simulation<static_build_tree>(rept, mal_node);
        //simulation<k_means_cluster<1> >(rept, mal_node);
        //simulation<k_means_cluster<2> >(rept, mal_node);
        simulation<k_means_cluster<4> >(rept, mal_node);
        simulation<block_p2p>(rept, mal_node);
        mal_node += 0.05;
    }
    return 0;
}
