#include "FileUtils.h"
#include "ExceptionManager.h"

#include <string>
#include <dirent.h>
#include "Exception.h"

using namespace std;

vector<NoteTransformer> FileUtils::transformers;

int FileUtils::getNumberOfFilesInDir(string directoryPath){
        DIR *dp;
        int i = 0;
        struct dirent *ep;     
        dp = opendir ("./");

        if (dp != NULL){
            while (ep = readdir (dp)){
                i++;
                }
        closedir (dp);
        }else{
        cerr << "Exception: the directory " + directoryPath + " could not been open.";
        }
        return i;
    }

void FileUtils::callRegisteredDestructors(){
    for (int i = 0; i < transformers.size(); i++){
        transformers.back().~NoteTransformer();
        transformers.pop_back();
    }
}

void FileUtils::registerTransformerToDestruct(NoteTransformer transformer){
    transformers.push_back(transformer);
}

void FileUtils::saveFloatMatrixToFiles(std::string fileName, float** matrix, int collums, int rows){
    ofstream outFile(fileName + ".txt");
        
    if (!outFile.is_open()) {
        throw Exception("The file " + fileName + " could not been opened", ExceptionType::FILE_HANDLEING);
    }

    outFile << collums << " " << rows;

    for (int i = 0; i < collums; i++){
        outFile << "\n";
        for (int j = 0; j < rows; j++){
            outFile << matrix[i][j] << " ";
        }
    }

    outFile.close();
}


void FileUtils::saveIntMatrixToFiles(std::string fileName, int** matrix, int collums, int rows){
    ofstream outFile(fileName + ".txt");
        
    if (!outFile.is_open()) {
        throw Exception("The file " + fileName + " could not been opened", ExceptionType::FILE_HANDLEING);
    }

    outFile << collums << " " << rows;

    for (int i = 0; i < collums; i++){
        outFile << "\n";
        for (int j = 0; j < rows; j++){
            outFile << matrix[i][j] << " ";
        }
    }

    outFile.close();
}

void FileUtils::saveFloatVectorToFiles(std::string fileName, float* vector, int rows){
    ofstream outFile(fileName + ".txt");
        
    if (!outFile.is_open()) {
        cerr << "yep";
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

    int collums, rows;
    inFile >> collums >> rows;

    float** matrix = new float*[collums];
    for (int i = 0; i < collums; ++i) {
        matrix[i] = new float[rows];
    }

    for (int i = 0; i < collums; ++i) {
        for (int j = 0; j < rows; ++j) {
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

    int collums, rows;
    inFile >> collums >> rows;

    int** matrix = new int*[collums];
    for (i = 0; i < collums; ++i) {
        matrix[i] = new int[rows];
    }

    for (i = 0; i < collums; ++i) {
        for (int j = 0; j < rows; ++j) {
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