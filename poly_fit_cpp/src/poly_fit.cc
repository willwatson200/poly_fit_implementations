
#include <iostream>
#include <vector>

#include "Eigen/Dense"
#include "cnpy/cnpy.h"

void compute_fit_coefficients(uint poly_order, const std::vector<double> &xs_data, const std::vector<double> &ys_observed, std::vector<double> &coefficients) {
    auto x_vec = Eigen::Map<const Eigen::VectorXd>(xs_data.data(), static_cast<int64_t>(xs_data.size()));
    auto y_vec = Eigen::Map<const Eigen::VectorXd>(ys_observed.data(), static_cast<int64_t>(ys_observed.size()));
    Eigen::MatrixXd vandermonde_matrix(xs_data.size(), poly_order + 1);
    vandermonde_matrix.col(0).setOnes();
    for (uint i = 0; i < poly_order; i++) {
        vandermonde_matrix.col(i+1).array() = vandermonde_matrix.col(i).array() * x_vec.array();
    }
    auto result_map = Eigen::Map<Eigen::VectorXd>(coefficients.data(), static_cast<int64_t>(coefficients.size()));
    result_map = vandermonde_matrix.householderQr().solve(y_vec);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
      std::cout << "Usage: specify poly_order, xs_path, ys_path   e.g. ./poly_fit 3 ../data" << std::endl;
      return -1;
    }
    uint poly_order = std::stoi(argv[1]);
    std::string data_path = argv[2];

    cnpy::NpyArray xs_arr = cnpy::npy_load(data_path + "/xs.npy");
    auto* loaded_data_xs = xs_arr.data<double>();
    std::vector<double> xs{};
    for (uint i=0; i < xs_arr.num_vals; i++) {
        xs.push_back(loaded_data_xs[i]);
    }

    cnpy::NpyArray ys_noise_arr = cnpy::npy_load(data_path + "/ys_noise.npy");
    auto* loaded_data_ys_noise = ys_noise_arr.data<double>();
    std::vector<double> ys_noise{};
    for (uint i=0; i < ys_noise_arr.num_vals; i++) {
        ys_noise.push_back(loaded_data_ys_noise[i]);
    }

    std::vector<double> coefficients(poly_order+1);
    compute_fit_coefficients(poly_order, xs, ys_noise, coefficients);

    cnpy::npy_save("cpp_poly_fit_coefficients.npy", coefficients.data(),{poly_order+1},"w");
}
