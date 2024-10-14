#ifndef MEMORYUTILS_H
#define MEMORYUTILS_H

class MemoryUtils{
    public:
     static void allocateMatrix(float**& matrix, int rows, int columns);
     static void allocateMatrixWithZeros(float**& matrix, int rows, int columns);
     static void fillMatrixWithZeros(float**& matrix, int rows, int columns);
};
#endif