#include <cmath>
#include <random>
#include <cstdio>
#include <algorithm>
#include <vector>

const double ERROR_LIMIT = 0.25;
const double ADDPATIVE_TIMESTEP = 0.25;
const double FLOAT_ZERO = 1e-6;
const double GRAVITY_RHO = 2e4;

const int NEAR_NEIGHBOR_NUM = 8;
const int RANDOM_NEIGHBOR_NUM = 16;


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
double distance(const EuclideanVector<D> &a, const EuclideanVector<D> &b) {
    EuclideanVector<D> c = a - b;
    return c.magnitude();
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

template<int D = 2> 
class VivaldiModel {
private: 
    Coordinate<D> local_coord;

public:
    std::vector<int> random_neighbor_set;
    bool have_enough_neighbor;

    VivaldiModel() {
        //Initialize the coordinate as the origin 
        //Set the height = 100 ms
        //Set the absolute error = 2

        local_coord = Coordinate<D>(EuclideanVector<D>(), 0, 2.0);

        have_enough_neighbor = false;
    }

    Coordinate<D> coordinate() {
        return local_coord;
    }

    EuclideanVector<D> vector() {
        return local_coord.vector();
    }

    // rtt -- round trip time (s) show()j
    void observe(int remote_id, Coordinate<D> remote_coord, double rtt) {
        if (have_enough_neighbor == false) {
            bool existed = false;
            for (auto id: random_neighbor_set) 
                if (id == remote_id) {
                    existed = true;
                    break;
                }
            if (existed == false) {
                random_neighbor_set.push_back(remote_id);
                if (random_neighbor_set.size() == RANDOM_NEIGHBOR_NUM)
                    have_enough_neighbor = true;
            }
        }
        if (rtt == 0) {
            printf("RTT = 0");
            return;
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

        if (new_error > local_coord.error()) {
            //printf("DRIFTING\n");
        }

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
            unit_v = v / predict_rtt;
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

        // avoid error
        if (new_height < 0) 
            new_height = 0;

        EuclideanVector<D> new_coord = local_coord.vector() + unit_v * weighted_force_magnitude;

        // Add Gravity force
        double new_coord_mag = new_coord.magnitude();
        EuclideanVector<D> unit_dir = new_coord / new_coord_mag;
        //double gravity_weight = new_coord_mag * new_coord_mag / GRAVITY_RHO;
        double gravity_weight = new_coord_mag / 500;
        new_coord = new_coord - unit_dir * gravity_weight;

        //Update the local coordinate
        local_coord = Coordinate<D>(new_coord, new_height, new_error);
    }
};




