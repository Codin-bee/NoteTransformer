#include "NoteTransformer.h"
#include <iostream>
#include <random>

using namespace std;

int main(){
    system("pause");
    NoteTransformer trans(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
    float** matrix = new float*[10];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < 10; i++){
        matrix[i] = new float[8];
        for (int j = 0; j < 8; j++){
           float randomValue = dis(gen);
            matrix[i][j] = randomValue;
        }
    }
    system("pause");
    trans.saveMatrixToFile("C:/Users/theco/cpp/MLProject/src/resources/myMatrix.txt", matrix, 10, 8);
    for (int i = 0; i < 10; i++)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
    return 0;
}