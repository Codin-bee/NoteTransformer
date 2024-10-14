#include "MemoryUtils.h"

void MemoryUtils::allocateMatrix(float**& matrix, int rows, int columns){
    matrix = new float*[rows];
    for (int i = 0; i < rows; i++){
        matrix[i] = new float[columns];
    }
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