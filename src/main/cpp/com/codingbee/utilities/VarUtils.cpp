#include "VarUtils.h"
#include "Exception.h"
#include <cmath>

int **VarUtils::floorAndCastToInt(float **matrix, int rows, int columns) {
    if (!matrix) {
        throw Exception("Input matrix is null", ExceptionType::INVALID_ARGUMENT);
    }

    int** outputMatrix = new int*[rows];

    for (int i = 0; i < rows; i++) {
        outputMatrix[i] = new int[columns];
        for (int j = 0; j < columns; j++) {
            outputMatrix[i][j] = floor(matrix[i][j]);
        }
    }

    return outputMatrix;
}

void VarUtils::copyMatrix(int **matrix1, int **&matrix2, int rows, int columns){
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < columns; j++){
            matrix2[i][j] = matrix1[i][j];
        }
    }
}

int VarUtils::getHighestIndexInSubVector(float *vector, int startingIndex, int endIndex){
    int highestIndex = startingIndex;
    int highestValue = 0;
    for (int i = startingIndex + 1; i <= endIndex; i++){
        if (vector[i] > highestValue){
            highestIndex = i;
            highestValue = vector[i];
        }
    }
    return highestIndex;
}
