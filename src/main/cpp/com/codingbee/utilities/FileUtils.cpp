#include "FileUtils.h"

#include <string>
#include <fstream>
#include <dirent.h>
#include "Exception.h"
#include <iostream>

using namespace std;

int FileUtils::getNumberOfFilesInDir(const string& directoryPath){
    DIR *dp;
    int fileCount = 0;
    struct dirent *ep;     
    dp = opendir(directoryPath.c_str());

    if (dp != NULL) {
        while ((ep = readdir(dp)) != NULL) {
            if (std::string(ep->d_name) != "." && std::string(ep->d_name) != "..") {
                fileCount++;
            }
        }
        closedir(dp);
    } else {
        throw Exception("Exception: the directory " + directoryPath + " could not be opened", ExceptionType::FILE_HANDLEING);
    }
    return fileCount;
}

void FileUtils::saveMatrixToFiles(std::string fileName, float** matrix, int rows, int columns){
    ofstream outFile(fileName + ".txt");
        
    if (!outFile.is_open()) {
        throw Exception("The file " + fileName + " could not been opened", ExceptionType::FILE_HANDLEING);
    }

    outFile << rows << " " << columns;

    for (int i = 0; i < rows; i++){
        outFile << "\n";
        for (int j = 0; j < columns; j++){
            outFile << matrix[i][j] << " ";
        }
    }

    outFile.close();
}

void FileUtils::saveMatrixToFiles(std::string fileName, int** matrix, int rows, int columns){
    ofstream outFile(fileName + ".txt");
        
    if (!outFile.is_open()) {
        throw Exception("The file " + fileName + " could not been opened", ExceptionType::FILE_HANDLEING);
    }

    outFile << rows << " " << columns;

    for (int i = 0; i < rows; i++){
        outFile << "\n";
        for (int j = 0; j < columns; j++){
            outFile << matrix[i][j] << " ";
        }
    }

    outFile.close();
}

void FileUtils::saveVectorToFiles(std::string fileName, float* vector, int rows){
    ofstream outFile(fileName + ".txt");
        
    if (!outFile.is_open()) {
        throw Exception("The file " + fileName + " could not been opened", ExceptionType::FILE_HANDLEING);
    }

    outFile << rows << "\n";

    for (int i = 0; i < rows; i++){
            outFile << vector[i] << " ";
    }

    outFile.close();
}

void FileUtils::saveVectorToFiles(std::string fileName, int* vector, int rows){
    ofstream outFile(fileName + ".txt");
        
    if (!outFile.is_open()) {
        throw Exception("The file " + fileName + " could not been opened", ExceptionType::FILE_HANDLEING);
    }

    outFile << rows << "\n";

    for (int i = 0; i < rows; i++){
            outFile << vector[i] << " ";
    }

    outFile.close();
}

float** FileUtils::readFloatMatrixFromFile(string fileName) {
    ifstream inFile(fileName + ".txt");

    if (!inFile.is_open()) {
        throw Exception("The file " + fileName + " could not been opened", ExceptionType::FILE_HANDLEING);
    }

    int columns, rows;
    inFile >> rows >> columns;

    float** matrix = new float*[rows];
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new float[columns];
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            inFile >> matrix[i][j];
        }
    }

    inFile.close();
    return matrix;
}

int** FileUtils::readIntMatrixFromFile(string fileName) {
    int i;
    ifstream inFile(fileName + ".txt");

    if (!inFile.is_open()) {
        throw Exception("The file " + fileName + " could not been opened", ExceptionType::FILE_HANDLEING);
    }

    int rows, columns;
    inFile >> rows >> columns;

    int** matrix = new int*[rows];
    for (i = 0; i < rows; ++i) {
        matrix[i] = new int[columns];
    }

    for (i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            inFile >> matrix[i][j];
        }
    }

    inFile.close();
    return matrix;
}

float* FileUtils::readFloatVectorFromFile(string fileName) {
    ifstream inFile(fileName + ".txt");

    if (!inFile.is_open()) {
        throw Exception("The file " + fileName + " could not been opened", ExceptionType::FILE_HANDLEING);
    }

    int rows;
    inFile >> rows;

    float* vector = new float[rows];
    for (int i = 0; i < rows; ++i) {
        inFile >> vector[i];
    }

    inFile.close();
    return vector;
}

int* FileUtils::readIntVectorFromFile(string fileName) {
    ifstream inFile(fileName + ".txt");

    if (!inFile.is_open()) {
        throw Exception("The file " + fileName + " could not been opened", ExceptionType::FILE_HANDLEING);
    }

    int rows;
    inFile >> rows;

    int* vector = new int[rows];
    for (int i = 0; i < rows; ++i) {
        inFile >> vector[i];
    }

    inFile.close();
    return vector;
}