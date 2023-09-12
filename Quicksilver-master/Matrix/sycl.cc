#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;

#define at(y, x, mat) (mat[y * n + x])

void matrix_multiplication(int n, int *a, int *b, int *c) {
    queue q;

    buffer<int, 1> a_buf(a, range<1>(n*n));
    buffer<int, 1> b_buf(b, range<1>(n*n));
    buffer<int, 1> c_buf(c, range<1>(n*n));

    q.submit([&](handler &h) {
        auto a_acc = a_buf.get_access<access::mode::read>(h);
        auto b_acc = b_buf.get_access<access::mode::read>(h);
        auto c_acc = c_buf.get_access<access::mode::write>(h);

        h.parallel_for< class matrix_mult >(range<2>(n, n), [=](id<2> idx) {
            int row = idx[0];
            int col = idx[1];
            int temp = 0;
            for (int k = 0; k < n; k++) {
                temp += at(row, k, a_acc) * at(k, col, b_acc);
            }
            at(row, col, c_acc) = temp;
        });
    }).wait();  // Wait for the kernel to finish execution before moving on
}

void print_matrix(int n, int *mat) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << at(i, j, mat) << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int n = 2000;
    std::vector<int> a(n*n), b(n*n), c(n*n);

    for (int i = 0; i < n*n; i++) {
        a[i] = i % 10;
        b[i] = (i % 10) + 1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Force CPU/GPU selection
    // sycl::device device = sycl::cpu_selector{}.select_device();
    sycl::device device = sycl::gpu_selector{}.select_device();

    matrix_multiplication(n, a.data(), b.data(), c.data());

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();

    std::cout << "Matrix multiplication took " << elapsed_time << " seconds." << std::endl;

    // std::cout << "Matrix A:" << std::endl;
    // print_matrix(n, a.data());
    // std::cout << "Matrix B:" << std::endl;
    // print_matrix(n, b.data());
    // std::cout << "Matrix C (Result of A * B):" << std::endl;
    // print_matrix(n, c.data());
    // Query and print device type, name, and number of compute units (cores)
    if (device.is_gpu()) {
        std::cout << "Using GPU device." << std::endl;
        std::cout << "Device Name: " << device.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "Number of Compute Units (Cores): " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    } else if (device.is_cpu()) {
        std::cout << "Using CPU device." << std::endl;
        std::cout << "Device Name: " << device.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "Number of Compute Units (Cores): " << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
    } else {
        std::cout << "Using unknown device type." << std::endl;
    }

    return 0;
}
