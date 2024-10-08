#ifndef MATHUTILS_H
#define MATHUTILS_H

class MathUtils{
    public:
    /// @brief multiplies two given matricies of the same size and than returns their product
    /// @param matrixA first matrix
    /// @param matrixB second matrix
    /// @param n size of the matricies
    /// @return the product of given matricies
    static float** multiplySameSquareMatricies(float** matrixA, float** matrixB, int n);


    /// @brief multiplies two given matricies and returns their product
    /// @param matrixA first matrix
    /// @param rowsA number of rows of the first matrix (the first index)
    /// @param columnsA number of columns in the first matrix (the second index)
    /// @param matrixB second matrix
    /// @param columnsB number of collumns in the second matrix (the second matrix)
    /// @return product of given matricies
    static float** multiplyMatricies(float** matrixA, int rowsA, int columnsA, float** matrixB, int columnsB);

    /// @brief Applies softmax function to given vector, which converts vector of real numbers into ptobability distribution
    /// @param vector Vector of real numbers
    /// @param vectorLength Length of the vector
    /// @param temperature The temperature variable for the function, if you do not know what you are doing keep it as 1
    static void applySoftmax(float* vector, int vectorLength, int temperature);

    static float leakyReLU(float n);
};
#endif