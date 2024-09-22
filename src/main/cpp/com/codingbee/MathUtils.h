#ifndef MATHUTILS_H
#define MATHUTILS_H

class MathUtils{
    /// @brief multiplies two given matricies of the same size and than returns their product
    /// @param matrixA first matrix
    /// @param matrixB second matrix
    /// @param n size of the matricies
    /// @return the product of given matricies
    static float** multiplySameSquareMatricies(float** matrixA, float** matrixB, int n);
};
#endif