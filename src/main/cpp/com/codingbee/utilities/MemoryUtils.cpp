#include "MemoryUtils.h"

void MemoryUtils::allocateMatrix(float**& matrix, int rows, int columns){
    matrix = new float*[rows];
    for (int i = 0; i < rows; i++){
        matrix[i] = new float[columns];
    }
}

void MemoryUtils::deallocateMatrix(float**& matrix, int rows){
    for (int i = 0; i < rows; i++){
        delete[] matrix;
    }
    delete[] matrix;
}

void MemoryUtils::deallocateMatrix(int**& matrix, int rows){
    for (int i = 0; i < rows; i++){
        delete[] matrix;
    }
    delete[] matrix;
}

void MemoryUtils::deallocate4DTensor(float****& tensor, int d1, int d2, int d3){
    for (int i = 0; i < d1; i++){
        for (int j = 0; j < d2; j++){
            for (int k = 0; k < d3; k++){
                delete[] tensor[i][j][k];
            }
            delete[] tensor[i][j];
        }
        delete[] tensor[i];
    }
    delete[] tensor;
}

void MemoryUtils::allocateMatrixWithZeros(float**& matrix, int rows, int columns){
    matrix = new float*[rows];
    for (int i = 0; i < rows; i++){
        matrix[i] = new float[columns]();
    }
}

void MemoryUtils::fillMatrixWithZeros(float**& matrix, int rows, int columns){
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            matrix[i][j] = 0;
        }
    }
    
}