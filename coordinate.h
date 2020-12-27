#include <cmath>
#include <random>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>

const double ERROR_LIMIT = 0.25;
const double ADDPATIVE_TIMESTEP = 0.25;
const double FLOAT_ZERO = 1e-6;
const double GRAVITY_RHO = 500;
const double MIN_ERROR = 0.1;
const double CENTROID_DRIFT = 25;

const int NEAR_NEIGHBOR_NUM = 8;
const int RANDOM_NEIGHBOR_NUM = 64;

#define sqr(x) ((x) * (x))


//D -- dimension
template<int D = 2>
class EuclideanVector {
public:
    double v[D];

    // Default Initializer
    EuclideanVector() {
        for (int i = 0; i < D; i++)
            v[i] = 0;
    }

    // Initialize from pre-defiened coordinates
    EuclideanVector(double *_v) {
        for (int i = 0; i < D; i++)
            v[i] = _v[i];
    }

    double magnitude() const {
        double ret = 0;
        for (int i = 0; i < D; i++)
            ret += v[i] * v[i];
        return sqrt(ret);
    }

    bool is_zero() const {
        for (int i = 0; i < D; i++)  {
            //printf("%.2f, %.2f\n", std::abs(v[i]), FLOAT_ZERO);
            if (std::fabs(v[i]) > FLOAT_ZERO) 
                return false;
        }
        return true;
    }

    void show() {
        for (int i = 0; i < D; i++)
            printf("%.2f ", v[i]);
    }
};

template<int D>
EuclideanVector<D> operator+(const EuclideanVector<D> &a, const EuclideanVector<D> &b) {
    EuclideanVector<D> c;
    for (int i = 0; i < D; i++)
        c.v[i] = a.v[i] + b.v[i];
    return c;
}

template<int D>
EuclideanVector<D> operator-(const EuclideanVector<D> &a, const EuclideanVector<D> &b) {
    EuclideanVector<D> c;
    for (int i = 0; i < D; i++)
        c.v[i] = a.v[i] - b.v[i];
    return c;
}

template<int D>
EuclideanVector<D> operator*(const EuclideanVector<D> &a, const double b) {
    EuclideanVector<D> c;
    for (int i = 0; i < D; i++)
        c.v[i] = a.v[i] * b;
    return c;
}

template<int D>
EuclideanVector<D> operator/(const EuclideanVector<D> &a, const double b) {
    if (b == 0) {
        printf("Divided by 0");
        return EuclideanVector<D>();
    }
    return a * (1.0 / b);
}

template<int D>
bool operator<(const EuclideanVector<D> &a, const EuclideanVector<D> &b) {
    return a.magnitude() < b.magnitude();
}

template<int D>
double distance(const EuclideanVector<D> &a, const EuclideanVector<D> &b) {
    EuclideanVector<D> c = a - b;
    return c.magnitude();
}

// a and b's dot product
template<int D>
double dot(const EuclideanVector<D> &a, const EuclideanVector<D> &b) {
    double ret = 0;
    for (int i = 0; i < D; i++)
        ret += a.v[i] * b.v[i];
    return ret;
}

double random_between_0_1() {
    return rand() / double(RAND_MAX);
}

template<int D> 
EuclideanVector<D> random_unit_vector() {
    for (;;) {
        double tmp[D];
        for (int i = 0; i < D; i++)
            tmp[i] = random_between_0_1();
        EuclideanVector<D> v(tmp);
        if (!v.is_zero()) {
            return v / v.magnitude();
        }
        //break;
    }
}

template<int D = 2> 
class Coordinate {
public:
    EuclideanVector<D> v;
    double h;
    double e;

    Coordinate(): v(), h(0), e(0) {}

    Coordinate(EuclideanVector<D> _v, double _h, double _e): 
        v(_v), h(_h), e(_e) {}

    EuclideanVector<D> vector() const {
        return v;
    }

    double height() const {
        return h;
    }

    double error() const {
        return e;
    }

    void show() const {
        printf("vector = (");
        for (int i = 0; i < D; i++)
            printf("%.2f, ", v.v[i]);
        printf("), height = %.2f, error = %.2f", h, e);
    }
};

template<int D> 
double estimate_rtt(const Coordinate<D> &a, const Coordinate<D> &b) {
    double ret = distance(a.vector(), b.vector()) + a.height() + b.height();
    return ret;
}

template<int D> 
double norm(const EuclideanVector<D> &a) {
    return a.magnitude();
}

double norm(const double &a) {
    return a;
}

template<typename T>
class HistoryStat {
private:
    size_t max_data_point;
    int history_counter;
    std::deque<T> q;
    //std::vector<T> old_data;
    //std::vector<T> loading_data;
    double median;
    double abs_median_dev;
    int UPDATE_ROUND;

public:
    //only update median per 50 observation 
    //static const int UPDATE_ROUND = std::min(50, max_data_point);
    HistoryStat(size_t _m) {
        max_data_point = _m;
        UPDATE_ROUND = std::min(50, int(max_data_point));
        history_counter = 0;
    }

    void observe(T new_data) {
        if (q.size() == max_data_point)
            q.pop_front();
        q.push_back(new_data);

        history_counter++;
        if (history_counter <= UPDATE_ROUND || history_counter % UPDATE_ROUND == 1) {
            std::vector<double> tmp;
            tmp.reserve(q.size());
            for (auto t: q) tmp.push_back(norm(t));
            std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
            median = tmp[tmp.size() / 2];

            abs_median_dev = 0;
            for (size_t i = 0; i < tmp.size(); i++)
                abs_median_dev += std::fabs(tmp[i] - median);
            abs_median_dev /= tmp.size();
        }
    }

    // the median may not be the latest one
    // it will update per 50 observation
    double get_median() {
        if (q.empty()) {
            return 0; 
        }
        return median;
    }

    double get_median_dev() {
        if (q.empty()) {
            return 0;
        }
        return abs_median_dev;
    }

    size_t collected_data_num() {
        return q.size();
    }

    bool is_full() {
        return q.size() == max_data_point;
    }

    void show() {
        printf("median = %.2f, dev = %.2f\n", median, abs_median_dev); 
        std::vector<double> tmp;
        for (auto t: q) tmp.push_back(norm(t));
        std::sort(tmp.begin(), tmp.end());
        printf("{");
        for (auto t: tmp) printf("%.2f, ", t);
        printf("}\n");
    }
};

template<int D = 2> 
class VivaldiModel {
private: 
    Coordinate<D> local_coord;
    HistoryStat<EuclideanVector<D> > history_force_stat;
    int self_id;
    int history_counter;
    bool enable_IN1;
    bool enable_IN2;
    bool enable_IN3;
    bool enable_gravity;

    std::unordered_map<int, Coordinate<D> > peer_coord;
    std::unordered_map<int, EuclideanVector<D> > received_force;
    std::unordered_map<int, HistoryStat<double> > received_rtt;
    std::unordered_set<int> blacklist;

public:
    std::unordered_set<int> random_peer_set;
    bool have_enough_peer;

    VivaldiModel(int id = 0): 
        local_coord(EuclideanVector<D>(), 0, 2.0),
        history_force_stat(100),
        self_id(id),
        history_counter(0),
        enable_IN1(true),
        enable_IN2(true),
        enable_IN3(true),
        enable_gravity(false),
        have_enough_peer(false) {
        //Initialize the coordinate as the origin 
        //Set the height = 100 ms
        //Set the absolute error = 2
    }

    Coordinate<D> coordinate() {
        return local_coord;
    }

    EuclideanVector<D> vector() {
        return local_coord.vector();
    }

    bool belong_to_peer_set(int id) {
        auto it = peer_coord.find(id);
        return it != peer_coord.end();
    }

    // rtt -- round trip time (s) show()j
    void observe(int remote_id, Coordinate<D> remote_coord, double rtt) {

        //First check blacklist!
        if (blacklist.find(remote_id) != blacklist.end()) {
            printf("blocked: self id = %d, remote id = %d\n", self_id, remote_id);
            return;
        }

        if (have_enough_peer == false) {
            if (!belong_to_peer_set(remote_id)) {
                peer_coord.emplace(remote_id, remote_coord);
                received_force.emplace(remote_id, EuclideanVector<D>());
                received_rtt.emplace(remote_id, HistoryStat<double>(10));

                random_peer_set.insert(remote_id);
                if (random_peer_set.size() == RANDOM_NEIGHBOR_NUM)
                    have_enough_peer = true;
            }
        }

        if (rtt == 0) {
            printf("RTT = 0");
            return;
        }

        // use last 10 observation's median to update the coordinate
        // more stable
        if (belong_to_peer_set(remote_id)) {
            auto it = received_rtt.find(remote_id);
            it -> second.observe(rtt);
            double med = it -> second.get_median();
            rtt = med;
            //if (rtt == 10000) {
            //    //printf("self_id = %d, remote_id = %d", self_id, remote_id);
            //}
        }

        // Sample weight balances local and remote error (1)
        //
        // 		w = ei/(ei + ej)
        //
        double weight = local_coord.error() / (local_coord.error() + remote_coord.error());

        // Compute relative error of this sample (2)
        //
        // 		es = | ||xi -  xj|| - rtt | / rtt
        //
        //EuclideanVector<D> diff_vec = local_coord.vector() - remote_coord.vector();
        //double diff_mag = diff_vec.magnitude();
        double predict_rtt = estimate_rtt(local_coord, remote_coord);
        double relative_error = std::fabs(predict_rtt - rtt) / rtt;
        //printf("relative_error = %.2f\n", relative_error);

        // Update weighted moving average of local error (3)
        //
        // 		ei = es × ce × w + ei × (1 − ce × w)
        //
        double weighted_error = ERROR_LIMIT * weight;
        double new_error = relative_error * weighted_error
            + local_coord.error() * (1.0 - weighted_error);

        if (new_error < MIN_ERROR)
            new_error = MIN_ERROR;

        //if (new_error > local_coord.error()) {
        //    //printf("DRIFTING\n");
        //}

        // Calculate the adaptive timestep (part of 4)
        //
        // 		δ = cc × w
        //
        double adaptive_timestep = ADDPATIVE_TIMESTEP * weight;

        // Weighted force (part of 4)
        //
        // 		δ × ( rtt − ||xi − xj|| )
        //
        // if rtt > predict_rtt
        //     too close, push away
        // if rtt < predict_rtt
        //     too far, pull together
        double weighted_force_magnitude = adaptive_timestep * (rtt - predict_rtt); 

        //if (rtt == 10000) {
            //printf("self_id = %d, w = %.2f\n", self_id, weighted_force_magnitude);
        //}

        // Unit vector (part of 4)
        //
        // 		u(xi − xj)
        //
        EuclideanVector<D> v = local_coord.vector() - remote_coord.vector();
        EuclideanVector<D> unit_v;
        if (v.is_zero()) {
            // if the coordinates are nearly the same, generate a random unit vector
            unit_v = random_unit_vector<D>();
            //unit_v = unit_v / (1 + local_coord.height() + remote_coord.height());
        } else {
            // calculate the unit vector (remote ---> self)
            //unit_v = v / v.magnitude();
            unit_v = v / predict_rtt;   //bug? v does not contain the height but predict_rtt contains the height
        }
        // Calculate the new height of the local node:
        //
        //      (Old height + remote.Height) * weighted_force / predict_rtt + old height
        //
        double new_height = local_coord.height();
        if (v.is_zero()) {
            new_height = local_coord.height() + weighted_force_magnitude;
        } else {
            new_height = local_coord.height() +
                (local_coord.height() + remote_coord.height()) * weighted_force_magnitude / predict_rtt;
        }

        // 		δ × ( rtt − ||xi − xj|| ) x u(xi - x)
        EuclideanVector<D> force = unit_v * weighted_force_magnitude;

        // avoid error
        if (new_height < 0) 
            new_height = 0;

        // IN3: decelaration rule
        double history_median = history_force_stat.get_median();
        double median_dev = history_force_stat.get_median_dev();

        //if (std::fabs(weighted_force_magnitude) > 100)
        //    return;

        if (enable_IN3 &&
            history_counter > 20 && 
            std::fabs(weighted_force_magnitude) > history_median + 5 * median_dev) { //history_force_stat.show();
            //printf("w = %.2f Violates IN3: decelaration rule, remote_id = %d\n", weighted_force_magnitude, remote_id);
            printf("Violates IN3\n");
            return;
        } else {
            if (rtt == 10000) {
                //history_force_stat.show();
                //printf("Fail to avoid outlier w = %.2f, median = %.2f, dev = %.2f, history cnt = %d\n", weighted_force_magnitude, history_median, median_dev, history_counter);
            }
        }

        if (enable_IN3)
            history_force_stat.observe(force);
        
        if (enable_IN1 && belong_to_peer_set(remote_id)) {
            auto it = received_force.find(remote_id);
            it -> second = it -> second + force;
        }

        EuclideanVector<D> new_coord = local_coord.vector() + force;

        if (enable_gravity){
            // Add Gravity force
            double new_coord_mag = new_coord.magnitude();
            EuclideanVector<D> unit_dir = new_coord / new_coord_mag;
            double gravity_weight = sqr(new_coord_mag / GRAVITY_RHO);
            //double gravity_weight = new_coord_mag / 500;
            new_coord = new_coord - unit_dir * gravity_weight;
        }
        

        //Update the local coordinate
        local_coord = Coordinate<D>(new_coord, new_height, new_error);

        history_counter++;
        //IN1: centroid rule
        if (enable_IN1 && history_counter > 20)
            security_in1();
    }
    void security_in1() {
        EuclideanVector<D> centroid;
        centroid = centroid + local_coord.vector();
        for (auto& it: peer_coord) 
            centroid = centroid + it.second.vector();
        centroid = centroid / (1.0 * peer_coord.size() + 1);

        if (centroid.magnitude() > CENTROID_DRIFT) {
            EuclideanVector<D> force;
            double max_force_mag = 0;
            int malicious_id = 0;

            for (auto& it: received_force) {
                double p = dot(it.second, centroid);
                if (p > max_force_mag) {
                    force = it.second;
                    max_force_mag = p;
                    malicious_id = it.first;
                }
            }

            printf("Violates In1\n");
            //printf("Violates In1, centroid mag = %.2f, self_id = %d, mal id = %d\n", centroid.magnitude(), self_id, malicious_id);
            EuclideanVector<D> new_coord = local_coord.vector() - force;
            double new_height = local_coord.height();
            double new_err = local_coord.error();
            local_coord = Coordinate<D>(new_coord, new_height, new_err);

            centroid.show();
            printf("\n");
            
            random_peer_set.erase(malicious_id);
            peer_coord.erase(malicious_id);
            received_force.erase(malicious_id);
            have_enough_peer = false;

            blacklist.insert(malicious_id);
        }
    }
};




