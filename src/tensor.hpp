#pragma once

#include <vector>
#include <iostream>
#include <stdexcept>

template <typename T>
class Tensor {
  public:
    std::vector<T> data;
    int rows;
    int cols;

    // We don't want to modify the underlying weights from the model
    Tensor(const std::vector<T>& values, int rows, int cols)
        : data(values), rows(rows), cols(cols) {}

    Tensor(int rows, int cols) 
        : data(rows * cols, T{}), rows(rows), cols(cols) {}

    void print() {
        for (auto r=0; r< rows; r++) {
            std::cout << "|";
            for (auto c=0; c < cols; c++) {
                std::cout << data[r * cols + c] << " ";
            }
            std::cout << "|\n";
        }
    }

    T& at(int row, int col) {
        return data[row * cols + col];
    }

    const T& at(int row, int col) const {
        return data[row * cols + col];
    }

};


template <typename T>
Tensor<T> add(const Tensor<T> &a, const Tensor<T> &b) {
    if (a.cols != b.cols || a.rows != b.rows)
        throw std::invalid_argument("Dimensions of matrices aren't the same");

    Tensor<T> c(a.rows, a.cols);
    const int size = a.rows * a.cols;
    for (int i = 0; i < size; i++)
        c.data[i] = a.data[i] + b.data[i];

    return c;
}

template <typename T>
Tensor<T> sub(const Tensor<T> &a, const Tensor<T> &b) {
    if (a.cols != b.cols || a.rows != b.rows)
        throw std::invalid_argument("Dimensions of matrices aren't the same");

    Tensor<T> c(a.rows, a.cols);
    const int size = a.rows * a.cols;
    for (int i = 0; i < size; i++)
        c.data[i] = a.data[i] - b.data[i];

    return c;
}

// Very naive and mat-mul implementation just to establish correctness baseline
template <typename T>
Tensor<T> mat_mul(const Tensor<T> &a, const Tensor<T> &b) {
    if (a.cols != b.rows)
        throw std::invalid_argument("Dimension of matrices are incompatible");

    Tensor<T> c(a.rows, b.cols);
    for (int i = 0; i < c.rows; i++) {
        for (int j=0; j < c.cols; j++) {
            for (int k=0; k < a.cols; k++) {
                c.data[i * c.cols + j] += a.data[i * a.cols + k] * b.data[k * b.cols + j];
            }
        }
    }
    return c;
}
