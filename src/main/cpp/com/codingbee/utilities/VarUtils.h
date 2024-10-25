#ifndef VARUTILS_H
#define VARUTILS_H

class TypeUtils{
    public:
    static int** floorAndCastToInt(float** matrix, int rows, int columns);
    static void copyMatrix(int** matrix1, int**& matrix2, int rows, int columns);
};

#endif