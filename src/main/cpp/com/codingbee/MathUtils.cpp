#include "MathUtils.h"
#include <math.h>
#include "Exception.h"

using namespace std;


float** MathUtils::multiplySameSquareMatricies(float** matrixA, float** matrixB, int n){
    if (n < 1){
        throw Exception("The matrix size has to be higher than 0", ExceptionType::INVALID_ARGUMENT);
    }
    if (matrixA == nullptr || matrixB == nullptr){
        throw Exception("The matricies can not be null pointers", ExceptionType::INVALID_ARGUMENT);
    }
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
    if(vectorLength < 1){
        throw Exception("The vector length has to be higher than 0", ExceptionType::INVALID_ARGUMENT);
    }
    if(temperature <= 0){
        throw Exception("Softmax temperature has to be higher than 0", ExceptionType::INVALID_ARGUMENT);
    }
    if(vector == nullptr){
        throw Exception("The vector cannot be a null pointer", ExceptionType::INVALID_ARGUMENT);
    }

    float maxVal = vector[0];
    for (int i = 1; i < vectorLength; i++) {
        if (vector[i] > maxVal) {
            maxVal = vector[i];
        }
    }

    float sum = 0;
    for (int i = 0; i < vectorLength; i++) {
        vector[i] = exp((vector[i] - maxVal) / temperature);
        sum += vector[i];
    }
    for (int i = 0; i < vectorLength; i++) {
        vector[i] = vector[i] / sum;
    }
}


float MathUtils::leakyReLU(float n){
    if (n < 0){
        return 0.01 * n;
    }
    return n;
}