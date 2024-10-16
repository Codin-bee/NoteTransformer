#ifndef MATHUTILS_H
#define MATHUTILS_H

class MathUtils{
    public:
    //Matricies

    /// @brief multiplies two given matricies of the same size and than returns their product
    /// @param matrixA first matrix
    /// @param matrixB second matrix
    /// @param n size of the matricies
    /// @return the product of given matricies
    /// @throws Invalid argument exception in case the matricies are null pointers or their size is passed as number lower than zero
    static float** multiplySameSquareMatricies(float** matrixA, float** matrixB, int n);

    /// @brief multiplies two given matricies and returns their product
    /// @param matrixA first matrix
    /// @param columnsA number of columns in the first matrix (the second index)
    /// @param rowsA number of rows of the first matrix (the first index)
    /// @param matrixB second matrix
    /// @param columnsB number of collumns in the second matrix (the second matrix)
    /// @return product of given matricies
    /// @throws Invalid argument exception in case the matricies are null pointers or any of their size is passed as number lower than zero
    static float** multiplyMatricies(float** matrixA, int rowsA, int columnsA, float** matrixB, int columnsB);


    //Activation functions

    /// @brief Applies softmax function to given vector, which converts vector of real numbers into ptobability distribution
    /// @param vector Vector of real numbers
    /// @param vectorLength Length of the vector
    /// @param temperature The temperature variable for the function, if you do not know what you are doing keep it as 1
    /// @throws Invalid argument exception in case the vector is null, the vector length is passed as number lower than zero
    ///         or the temperature is equal to or lower than zero
    static void applySoftmax(float*& vector, int vectorLength, int temperature);

    /// @brief Applies leaky ReLU activation function to given number
    /// @param n The number to activate
    /// @return The activated value
    static float leakyReLU(float n);

    /// @brief Applies sigmoid activation function to given number
    /// @param n The number to activate
    /// @return The activated value
    static float sigmoid(float n);


    //Vectors

    /// @brief Adds elements of two vectors at each index and returns the sum in new vector
    /// @param vectorA First vector
    /// @param vectorB Second vector
    /// @param vectorLength Length of the vectors
    /// @return New vector containing the sums of given two vectors
    /// @throws Invalid argument exception, in case the vectors are null pointers or their length is zero or lower
    static float* addVectors(float* vectorA, float* vectorB, int vectorLength);

    /// @brief Multiplies elements of two vectors at each index and returns the product in new vector
    /// @param vectorA First vector
    /// @param vectorB Second vector
    /// @param vectorLength Length of the vectors
    /// @return New vector containing the products of given two vectors
    /// @throws Invalid argument exception, in case the vectors are null pointers or their length is zero or lower
    static float* multiplyVectors(float* vectorA, float* vectorB, int vectorLength);

    /// @brief Adds all elements of given vector and returns their sum
    /// @param vector The vector to sum
    /// @param vectorLength Length of the vector
    /// @return The sum of all elements in given vector
    /// @throws Invalid argument exception, in case the vector is null pointer or its length is zero or lower
    static float addVectorElements(float* vector, int vectorLength);
};
#endif