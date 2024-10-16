#ifndef UTILITIES_H
#define UTILITIES_H

#include <string>
#include "NoteTransformer.h"

class FileUtils{
    public:
        //General file functions

        /// @brief Searches files in given directory and returns their total cout
        /// @param directoryPath The directory to be searched
        /// @return Number of files in given directory
        /// @throws File managing Exception if the directory can not be accesed
        static int getNumberOfFilesInDir(const std::string& directoryPath);


        //Float matrix

        /// @brief Saves given matrix to given file
        /// @param fileName The file name or file path to the file where matrix will be saved
        /// @param matrix The matrix containing values to be saved
        /// @param rows Number of rows in the matrix
        /// @param columns Number of columns in given matrix
        /// @throws File managing Exception if the file can not be opened
        static void saveMatrixToFiles(std::string fileName, float** matrix, int rows, int columns);

        /// @brief Reads the float matrix from given directory
        /// @param fileName The name of file to read the values from or the path to it
        /// @return The matrix containing values from the file
        /// @throws File managing Exception if the file can not be opened
        static float** readFloatMatrixFromFile(std::string fileName);


        //Int matrix

        /// @brief Saves given matrix to given file
        /// @param fileName The file name or file path to the file where matrix will be saved
        /// @param matrix The matrix containing values to be saved
        /// @param rows Number of rows in the matrix
        /// @param columns Number of columns in given matrix
        /// @throws File managing Exception if the file can not be opened
        static void saveMatrixToFiles(std::string fileName, int** matrix, int rows, int columns);

        /// @brief Reads the values from given file
        /// @param fileName The name of the file or path to it
        /// @return The matrix containing the values from the file
        /// @throws File managing Exception if the file can not be opened
        static int** readIntMatrixFromFile(std::string fileName);


        //Float vector
        
        /// @brief Saves given float vector to file.
        /// @param fileName Name of the created file or the full path
        /// @param vector The vector to save
        /// @param rows Length of given vector
        /// @throws File managing Exception if the file can not be opened
        static void saveFloatVectorToFiles(std::string fileName, float* vector, int rows);

        /// @brief Reads vector from given file
        /// @param fileName The path to the file
        /// @return The vector containing values from the file
        /// @throws File managing Exception if the file can not be opened
        static float* readFloatVectorFromFile(std::string fileName);
};

#endif