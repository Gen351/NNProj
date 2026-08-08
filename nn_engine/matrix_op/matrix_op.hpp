#pragma once

#include<stdexcept>
#include<time.h>

#include<vector>
#include<math.h>

#include "matrix.hpp"

namespace MatrixOp {

    /* Functions using the structs */
    /// @brief x Matrix
    /// @param v - Multidimentional std::vector
    /// @param m - Multidimentional Matrix
    /// @return std::vector<T> with Dimensions 1 x M.cols()
    /// @warning v.size() must equal m.rows()
    template<typename T>
    std::vector<T> multiply(const std::vector<T>& v, const Matrix<T>& M) {
        #ifndef NDEBUG
            if(v.size() != M.rows()) { throw std::runtime_error("v x M: Dimensions Mismatch!");}
        #endif

        std::vector<T> vxM(M.cols());
        for(size_t i = 0; i < M.cols(); i++) {
            T sum = 0;
            // M[j][i]: safe because (v.size() == M.rows())
            for(size_t j = 0; j < v.size(); j++) sum += v[j] * M[j][i];
            vxM[i] = sum;
        }
        return vxM;
    }

    /// @brief  Matrix x Matrix
    /// @param A - matrix1
    /// @param B - matrix2
    /// @return Matrix<T> with Dimensions A.rows() x B.cols()
    /// @attention Dimensions from: https://i0.wp.com/www.mathbootcamps.com/wp-content/uploads/not-commutative.jpg?resize=520%2C463
    /// @warning optimizatio because of the flat matrix implementation
    /*
        By swapping the j and k loops (known as the IKJ pattern), 
            the innermost loop now moves across the rows of Matrix B. 
            Since the memory is flat, 
            **B[k][0], B[k][1], B[k][2]** are all right next to each other. 
            This makes your CNN significantly faster because the CPU can pre-fetch the data.
    */
    template<typename T>
    Matrix<T> multiply(const Matrix<T>& A, const Matrix<T>& B) {
        #ifndef NDEBUG
            if(A.cols() != B.rows()) { throw std::runtime_error("A x B: Dimensions Mismatch!");}
        #endif

        Matrix<T> AxB(A.rows(), B.cols(), T{});
        for(size_t i = 0; i < A.rows(); i++) {
            for(size_t k = 0; k < A.cols(); k++) {

                T temp = A[i][k];
                for(size_t j = 0; j < B.cols(); j++) {
                    AxB[i][j] += temp * B[k][j];
                }
            }
        }
        return AxB;
    }

    /// @brief 
    /// @param a - std::vector a of dimention(d): 1 x n
    /// @param b - std::vector b of dimention(d): 1 x n
    /// @return T scalar dot products of std::vector a * b
    template<typename T>
    T dot(const std::vector<T>& a, const std::vector<T>& b) {
        #ifndef NDEBUG
            if(a.size() != b.size()) { throw std::runtime_error("a dot b: Dimensions Mismatch!"); }
        #endif

        T sum = 0;
        for(size_t i = 0; i < a.size(); i++) sum += a[i] * b[i];
        return sum;
    }

    /// @brief 
    /// @param A - matrix1  
    /// @param B - matrix2
    /// @return T scalar dot products of matrix A * B
    /// @warning ni Gemini for CNNs (Technical bs)
    /*  
        The **"CNN Case"**: Frobenius Inner ProductIn a CNN, 
            when a Kernel (Filter) slides over an Input Image, 
            you take a small patch of the image (**A**) and the kernel (**B**). 
            You multiply every corresponding pixel and sum them all up 
            into one single value for the next layer's pixel.
            
            This is technically called the **Frobenius Inner Product**. 
            If this is what you want your dot function to do, 
            it would look like this:
    */
    template<typename T>
    T dot(const Matrix<T>& A, const Matrix<T>& B) {
        #ifndef NDEBUG
            if(A.cols() != B.cols() || A.rows() != B.rows()) { throw std::runtime_error("A dot B: Dimensions Mismatch!"); }
        #endif

        T sum = 0;
        for(size_t i = 0; i < A.data.size(); i++)
            sum += A.data[i] * B.data[i];
        return sum;
    }


    template<typename T>
    Matrix<T> add(const Matrix<T>& A, const Matrix<T>& B) {
        #ifndef NDEBUG
        if(A.rows() != B.rows() || A.cols() != B.cols()) { throw std::runtime_error("A + B: Dimensions Mismatch!"); }
        #endif

        Matrix<T> res(A.rows(), A.cols());

        for(size_t i = 0; i < A.data.size(); i++) {
            res.data[i] = A.data[i] + B.data[i]; 
        }

        return res;
    }

    /// @brief (cum[i][j] = cum[i][j] + source[i][j])
    /// @param cum = Matrix<T> cumulative, this is where you put your result 
    /// @param source = addend
    template<typename T>
    void addInPlace(Matrix<T>& cum, const Matrix<T>& source) {
        #ifndef NDEBUG
            if(cum.rows() != source.rows() || cum.cols() != source.cols()) { throw std::runtime_error("cum + source: Dimensions Mismatch!"); }
        #endif

        for(size_t i = 0; i < cum.data.size(); i++) {
            cum.data[i] +=  source.data[i]; 
        }
    }


    /// @example 
    /*
        A = 
            | 1 2 3 |
            | 4 5 6 |
            | 7 8 9 |
        transpose(A) =
            | 1 4 7 |
            | 2 5 8 |
            | 3 6 9 |

        B = 
            | 1 2 3 4 5 |
            | 6 7 8 9 1 |
            | 2 3 4 5 6 |
        transpose(B) = 
            | 1 6 2 |
            | 2 7 3 |
            | 3 8 4 |
            | 4 9 5 |
            | 5 1 6 |
    */
    /// @param A - Matrix (n x m) 
    /// @return A^T - Matrix (m x n)
    template<typename T>
    Matrix<T> transpose(const Matrix<T>& A) {
        Matrix<T> transposed(A.cols(), A.rows());

        for(size_t i = 0; i < A.cols(); i++) {
            for(size_t j = 0; j < A.rows(); j++) {
                transposed[i][j] = A[j][i];
            }
        }

        return transposed;
    }


    /// @brief flip a matrix on the vertical axis
    /// @return void, flips the referenced Matrix 
    template<typename T>
    void flipMatrixRef(Matrix<T>& M) {
        for(size_t j = 0; j < M.cols() / 2; j++) {
            for(size_t i = 0; i < M.rows(); i++) {
                std::swap(M[i][j], M[i][M.cols() - j - 1]);
            }
        }
    }


    /// @brief   
    /// @param shiftDir {0=[<], 1=[^], 2=[>], 3=[v]}
    /// @param pad padding for the shift 
    /// @return shifted Matrix
    template<typename T>
    Matrix<T> shift(const Matrix<T>& M, int shiftDir, int shiftAmount, T pad) {
        Matrix<T> shifted(M.rows(), M.cols(), pad);
        
        // Offset values
        int di = 0, dj = 0;
        shiftDir %= 4;
        shiftAmount = shiftAmount > 2  || shiftAmount < 1 ? 1 : shiftAmount;

        if (shiftDir == 0) dj = -shiftAmount;      // Left
        else if (shiftDir == 1) di = -shiftAmount; // Up
        else if (shiftDir == 2) dj = shiftAmount;  // Right
        else if (shiftDir == 3) di = shiftAmount;  // Down

        for (int i = 0; i < (int)M.rows(); i++) {
            for (int j = 0; j < (int)M.cols(); j++) {
                int src_i = i - di;
                int src_j = j - dj;

                // Only copy if the source pixel is within the original bounds
                if (src_i >= 0 && src_i < (int)M.rows() && src_j >= 0 && src_j < (int)M.cols()) {
                    shifted[i][j] = M(src_i, src_j); 
                }
                // Otherwise, it stays as the 'pad' value initialized in the constructor
            }
        }
        return shifted;
    }



    /// @brief rotate a Matrix
    /// @return rotated Matrix
    template<typename T>
    Matrix<T> rotate180(const Matrix<T>& M) {
        Matrix<T> rotated(M.rows(), M.cols());
        for(size_t i = 0; i < M.rows(); i++) {
            for(size_t j = 0; j < M.cols(); j++) {
                rotated[i][j] = M[M.rows() - i - 1][M.cols() - j - 1];
            }
        }

        return rotated;
    }

    /// @brief pad a Matrix with 0
    /// @example
    /// padSize = 1 
    /*  M = 
                | 1 3 5 7 |
                | 2 5 7 9 |
                | 1 1 4 1 |
    */
    /* @return padded = 
                | 0 0 0 0 0 0 |
                | 0 1 3 5 7 0 |
                | 0 2 5 7 9 0 |
                | 0 1 1 4 1 0 |
                | 0 0 0 0 0 0 |
    */
    template<typename T>
    Matrix<T> pad(const Matrix<T> M, size_t padSize) {
        Matrix<T> padded(M.rows() + padSize * 2, M.cols() + padSize * 2, T{0});

        for(size_t i = 0; i < M.rows(); i++) {
            for(size_t j = 0; j < M.cols(); j++) {
                padded[i + padSize][j + padSize] = M[i][j];
            }
        }
        return padded;
    }



    /*============================================
    ==============================================
    >> C  N  N    O  P  E  R  A  T  I  O  N  S  <<
    ==============================================
    ============================================*/
    
    /// @brief Converts a Matrix into a "flat" std::vector or a 1-D array
    /// @brief ... using std::std::vector's assignment operator. 
    /// @param A - Matrix (n x m)
    /// @return V - std::vector (n * m)
    template<typename T>
    std::vector<T> flatten(const Matrix<T>& A) {
        return A.data;
    }

    float initRandFloat() {    
        return ((float)rand()/(float)RAND_MAX)
                - ((float)rand()/(float)RAND_MAX);
    }
    int initRandInt(int max=RAND_MAX) {
        return rand()/max;
    }

    float XavierGlorotRandom(const int inputs) {
        return initRandFloat() * std::sqrt(1.0f / inputs);
    }

}