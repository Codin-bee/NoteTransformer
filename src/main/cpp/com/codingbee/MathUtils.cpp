#include "MathUtils.h"

float** MathUtils::multiplySameSquareMatricies(float** matrixA, float** matrixB, int n){
    float ** matrixC = new float*[n];
        for (int i = 0; i < n; i++){
            matrixC[i] = new float[n];
            for (int k =0; k < n; k++){
                for (int j = 0; j < n; j++)
                {
                    matrixC[i][j] = matrixA[i][k] * matrixB[k][j];
                }
            }
        } 
    return matrixC;
}