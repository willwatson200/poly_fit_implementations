import numpy as np
import sys
import os.path

def generate_data(n_points, n_order, noise_factor):
    xs = np.float64(np.sort(5 * (2*np.random.rand(n_points) - 1)))
    coefficients = np.float64(10* (2*np.random.random(n_order+1) - 1))

    poly = np.poly1d(coefficients)
    ys = np.float64(poly(xs))
    ys_noise = np.float64(ys + noise_factor*np.random.randn(n_points))

    with open(os.path.dirname(__file__) + '/../data/xs.npy', 'wb') as f:
        np.save(f, xs)
    with open(os.path.dirname(__file__) + '/../data/coefficients.npy', 'wb') as f:
        np.save(f,  coefficients)
    with open(os.path.dirname(__file__) + '/../data/ys_noise.npy', 'wb') as f:
        np.save(f,  ys_noise)

if __name__ =='__main__':
    argv = sys.argv
    if len(argv) !=4:
        print("Usage: specify\nnum_points\nnoise factor")
        sys.exit()

    N_POINTS = int(argv[1])
    N_ORDER = int(argv[2])
    NOISE_FACTOR = int(argv[3])
    
    generate_data(N_POINTS, N_ORDER, NOISE_FACTOR)
       