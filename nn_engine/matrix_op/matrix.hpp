#pragma once

#include<stdexcept>
#include<vector>


template<typename T>
// @brief Init with (r, c) row_size, col_size
struct Matrix {
    size_t row;
    size_t col;

    std::vector<T> data;

    Matrix() = default;

    Matrix(const Matrix& other) = default;
    Matrix& operator=(const Matrix& other) = default;

    Matrix(Matrix&& other) noexcept = default;
    Matrix& operator=(Matrix&& other) noexcept = default;

    // init with (r, c) row_size, col_size
    Matrix(size_t r, size_t c, T init = T{}) 
        : row(r), col(c), data(r * c, init)
    {}

    // init with (r, c) row_size, col_size
    Matrix(size_t r, size_t c, std::vector<T> init)
        : row(r), col(c), data(init)
    {}


    T* operator[](size_t r) {
        #ifndef NDEBUG
            if(r >= row) { throw std::runtime_error("Matrix[]: Invalid Indexing!"); }
        #endif
        return &data[r * col];
    }
    const T* operator[](size_t r) const {
        #ifndef NDEBUG
            if(r >= row) { throw std::runtime_error("Matrix[]: Invalid Indexing!"); }
        #endif
        return &data[r * col];
    }

    T& operator()(size_t r, size_t c) {
        #ifndef NDEBUG
            if(r >= row || c >= col) { throw std::runtime_error("Matrix[][]: Invalid Indexing"); }
        #endif
        return data[r * col + c];
    }
    const T& operator()(size_t r, size_t c) const {
        #ifndef NDEBUG
            if(r >= row || c >= col) { throw std::runtime_error("Matrix[][]: Invalid Indexing"); }
        #endif
        return data[r * col + c];
    }

    // Get contigous data
    const auto begin() const {
        return data.begin();
    }
    const auto end() const {
        return data.end();
    }

    auto begin() {
        return data.begin();
    }
    auto end() {
        return data.end();
    }

    // Get contiguous row
    auto row_begin(size_t r_indx) {
        #ifndef NDEBUG
            if (r_indx >= row) { throw std::runtime_error("row_begin: Out of bounds"); }
        #endif
        return data.begin() + (r_indx * col);
    }

    auto row_end(size_t r_indx) {
        #ifndef NDEBUG
            if (r_indx >= row) { throw std::runtime_error("row_end: Out of bounds"); }
        #endif
        return data.begin() + ((r_indx + 1) * col);
    }

    // Const overloads (for const matrices)
    auto row_begin(size_t r_indx) const {
        #ifndef NDEBUG
            if (r_indx >= row) { throw std::runtime_error("row_begin: Out of bounds"); }
        #endif
        return data.cbegin() + (r_indx * col);
    }

    auto row_end(size_t r_indx) const {
        #ifndef NDEBUG
            if (r_indx >= row) { throw std::runtime_error("row_end: Out of bounds"); }
        #endif
        return data.cbegin() + ((r_indx + 1) * col);
    }

    size_t rows() const { return row; }
    size_t cols() const { return col; }
};
