#include "MathUtils.h"
#include <math.h>

using namespace std;


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

void MathUtils::applySoftmax(float* vector, int vectorLength, int temperature){
        float sum = 0;
        int i;
        for (i = 0; i < vectorLength; i++)
        {
            sum += exp(vector[i] / temperature);
        }
        for (i = 0; i < vectorLength; i++)
        {
            vector[i] = vector[i] / sum;
        }
    }