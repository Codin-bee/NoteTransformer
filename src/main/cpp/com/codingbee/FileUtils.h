#ifndef UTILITIES_H
#define UTILITIES_H

#include <string>
#include "NoteTransformer.h"

class FileUtils{
    public:
        static int getNumberOfFilesInDir(std::string directoryPath);
        static void registerTransformerToDestruct(NoteTransformer transformer);
        static void callRegisteredDestructors();
    private:
        static vector<NoteTransformer> transformers;
};

#endif