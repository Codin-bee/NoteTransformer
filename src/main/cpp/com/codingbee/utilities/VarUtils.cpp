#include "VarUtils.h"
#include <math.h>

int **TypeUtils::floorAndCastToInt(float **matrix, int rows, int columns){
    int** outputMatrix = new int*[rows];
    for (int i = 0; i < rows; i++){
        outputMatrix[i] = new int[columns];
        for (int j = 0; j < columns; j++){
            outputMatrix[i][j] = floor(matrix[i][j]);
        }
    }
    return outputMatrix;
}

void TypeUtils::copyMatrix(int **matrix1, int **&matrix2, int rows, int columns){
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            matrix2[i][j] = matrix1[i][j];
        }
    }
}