#ifndef UTILITIES_H
#define UTILITIES_H

#include <string>
#include "NoteTransformer.h"

class FileUtils{
    public:
        static int getNumberOfFilesInDir(std::string directoryPath);
        static void registerTransformerToDestruct(NoteTransformer transformer);
        static void callRegisteredDestructors();
        static void saveFloatMatrixToFiles(const string& fileName, float** matrix, int collums, int rows);
        static void saveIntMatrixToFiles(const string& fileName, int** matrix, int collums, int rows);
        static float** readFloatMatrixFromFile(std::string fileName);
        static int** readIntMatrixFromFile(std::string fileName);
    private:
        static vector<NoteTransformer> transformers;
};

#endif