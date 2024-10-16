#include "MathUtils.h"
#include <math.h>
#include "Exception.h"
#include <iostream>

using namespace std;


float** MathUtils::multiplySameSquareMatricies(float** matrixA, float** matrixB, int n){
    if (n < 1){
        throw Exception("The matrix size has to be higher than 0", ExceptionType::INVALID_ARGUMENT);
    }
    if (matrixA == nullptr || matrixB == nullptr){
        throw Exception("None of the matricies can be a null pointer", ExceptionType::INVALID_ARGUMENT);
    }

    float **matrixC = new float*[n];
    for (int i = 0; i < n; i++) {
        matrixC[i] = new float[n];
        for (int j = 0; j < n; j++) {
            matrixC[i][j] = 0;
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
    return matrixC;
}

float** MathUtils::multiplyMatricies(float** matrixA, int rowsA, int columnsA, float** matrixB, int columnsB){
    if (columnsA < 1 || columnsB < 1 || rowsA < 1){
        throw Exception("All the matrix sizes have to be higher than 0", ExceptionType::INVALID_ARGUMENT);
    }
    if (matrixA == nullptr || matrixB == nullptr){
        throw Exception("None of the matricies can be a null pointer", ExceptionType::INVALID_ARGUMENT);
    }

    float** matrixC = new float*[rowsA];
    for (int i = 0; i < rowsA; i++) {
        matrixC[i] = new float[columnsB];
        for (int j = 0; j < columnsB; j++) {
            matrixC[i][j] = 0;
        }
    }
    
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < columnsB; j++) {
            for (int k = 0; k < columnsA; k++) {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }
    return matrixC;
}

void MathUtils::applySoftmax(float*& vector, int vectorLength, int temperature){
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
        return 0.01f * n;
    }
    return n;
}

float MathUtils::sigmoid(float n){
    return 1 / (1 + exp(-n));
}

float MathUtils::addVectorElements(float* vector, int vectorLength){
    if(vector == nullptr){
        throw Exception("The vector can not be a null pointer", ExceptionType::INVALID_ARGUMENT);
    }
    if(vectorLength <= 0){
        throw Exception("The vector lenght can not be a zero or lower", ExceptionType::INVALID_ARGUMENT);
    }
    float sum = 0;
    for (int i = 0; i < vectorLength; i++){
        sum += vector[i];
    }
    return sum;
}

float* MathUtils::addVectors(float* vectorA, float* vectorB, int vectorLength){
    if (vectorA == nullptr || vectorB == nullptr){
        throw Exception("The vectors can not be null pointers", ExceptionType::INVALID_ARGUMENT);
    }
    if (vectorLength <= 0){
        throw Exception("The vector length can not be zero or lower", ExceptionType::INVALID_ARGUMENT);
    }
    float* vectorC = new float[vectorLength];
    for (int i = 0; i < vectorLength; i++){
        vectorC[i] = vectorA[i] + vectorB[i];
    }
    return vectorC;
}

float* MathUtils::multiplyVectors(float* vectorA, float* vectorB, int vectorLength){
    if (vectorA == nullptr || vectorB == nullptr){
        throw Exception("The vectors can not be null pointers", ExceptionType::INVALID_ARGUMENT);
    }
    if (vectorLength <= 0){
        throw Exception("The vector length can not be zero or lower", ExceptionType::INVALID_ARGUMENT);
    }
    float* vectorC = new float[vectorLength];
    for (int i = 0; i < vectorLength; i++){
        vectorC[i] = vectorA[i] * vectorB[i];
    }
    return vectorC;
}