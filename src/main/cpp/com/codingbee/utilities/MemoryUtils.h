#ifndef MEMORYUTILS_H
#define MEMORYUTILS_H

class MemoryUtils{
    public:

     /// @brief Allocates matrix of given sizes at the given adress
     /// @param matrix The pointer to the matrix adress
     /// @param rows Number of rows in the matrix
     /// @param columns Number of columns in the matrix
     static void allocateMatrix(float**& matrix, int rows, int columns);

     /// @brief Allocates matrix of given sizes at the given adress, where all the values are set to 0
     /// @param matrix The pointer to the matrix adress
     /// @param rows Number of rows in the matrix
     /// @param columns Number of columns in the matrix
     static void allocateMatrixWithZeros(float**& matrix, int rows, int columns);

     /// @brief Deallocates matrix at given adress with given size
     /// @param matrix The pointer to the matrix adress
     /// @param rows Number of rows in the matrix
     static void deallocateMatrix(float**& matrix, int rows);

     /// @brief Deallocates matrix at given adress with given size
     /// @param matrix The pointer to the matrix adress
     /// @param rows Number of rows in the matrix
     static void deallocateMatrix(int**& matrix, int rows);

     /// @brief Deallocates 4D tensor at given adress with given size
     /// @param tensor The pointer to the tensor adress
     /// @param d1 First dimension of the tensor
     /// @param d2 Second dimension of the tensor
     /// @param d3 Third dimension of the tensor
     static void deallocate4DTensor(float****& tensor, int d1, int d2, int d3);

     /// @brief Fill matrix at given adress with zeros
     /// @param matrix The pointer to given matrix
     /// @param rows Number of rows in the matrix
     /// @param columns Number of columns in the matrix
     static void fillMatrixWithZeros(float**& matrix, int rows, int columns);
};
#endif