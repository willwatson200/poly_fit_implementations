
#include <iostream>
#include <vector>

#include "Eigen/Dense"
#include "cnpy/cnpy.h"

std::vector<double> compute_fit_coefficients(uint poly_order, const std::vector<double> &xs_data, const std::vector<double> &ys_observed) {
    auto x_vec = Eigen::Map<const Eigen::VectorXd>(xs_data.data(), static_cast<int64_t>(xs_data.size()));
    auto y_vec = Eigen::Map<const Eigen::VectorXd>(ys_observed.data(), static_cast<int64_t>(ys_observed.size()));
    Eigen::MatrixXd vandermonde_matrix(xs_data.size(), poly_order + 1);
    vandermonde_matrix.col(0).setOnes();
    for (uint i = 0; i < poly_order; i++) {
        vandermonde_matrix.col(i+1).array() = vandermonde_matrix.col(i).array() * x_vec.array();
    }
    std::vector<double> result(poly_order + 1);
    auto result_map = Eigen::Map<Eigen::VectorXd>(result.data(), static_cast<int64_t>(result.size()));
    result_map = vandermonde_matrix.householderQr().solve(y_vec);
    return result;
}

double eval_polynomial(const std::vector<double> &coeffs, double x_val) {
    double result{};
    double x_p = 1.0;
    for (auto coeff: coeffs) {
        result += x_p * coeff;
        x_p *= x_val;
    }
    return result;
}

int main() {
    cnpy::NpyArray xs_arr = cnpy::npy_load("../../data/xs.npy");
    auto* loaded_data_xs = xs_arr.data<double>();
    std::vector<double> xs{};
    for (uint i=0; i < xs_arr.num_vals; i++) {
        xs.push_back(loaded_data_xs[i]);
    }

    cnpy::NpyArray ys_noise_arr = cnpy::npy_load("../../data/ys_noise.npy");
    auto* loaded_data_ys_noise = ys_noise_arr.data<double>();
    std::vector<double> ys_noise{};
    for (uint i=0; i < ys_noise_arr.num_vals; i++) {
        ys_noise.push_back(loaded_data_ys_noise[i]);
    }

    std::vector<double> coeffs = compute_fit_coefficients(3, xs, ys_noise);
    std::cout << "coefficients" << std::endl;
    
    for (auto &coeff: coeffs) {
        std::cout << coeff << std::endl;
    }
}