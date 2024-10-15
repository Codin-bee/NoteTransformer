#ifndef MEMORYUTILS_H
#define MEMORYUTILS_H

class MemoryUtils{
    public:
     static void allocateMatrix(float**& matrix, int rows, int columns);
     static void allocateMatrixWithZeros(float**& matrix, int rows, int columns);
     static void deallocateMatrix(float**& matrix, int rows);
     static void deallocate4DTensor(float****& tensor, int d1, int d2, int d3);
     static void fillMatrixWithZeros(float**& matrix, int rows, int columns);
};
#endif