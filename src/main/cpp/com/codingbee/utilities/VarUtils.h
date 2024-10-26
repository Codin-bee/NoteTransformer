#ifndef VARUTILS_H
#define VARUTILS_H

class VarUtils{
    public:
    static int** floorAndCastToInt(float** matrix, int rows, int columns);
    static void copyMatrix(int** matrix1, int**& matrix2, int rows, int columns);
    static int getHighestIndexInSubVector(float** vector, int startingIndex, int endIndex);
};

#endif