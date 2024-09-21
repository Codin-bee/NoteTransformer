#include "FileUtils.h"
#include "ExceptionManager.h"

#include <string>
#include <dirent.h>
#include <iostream>

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

void FileUtils::saveMatrixToFiles(const string& fileName, float** matrix, int collums, int rows){
        ofstream outFile(fileName);
        
    if (!outFile.is_open()){
        cerr << "Exception: the file " + fileName + "could not been open.";
        ExceptionManager::processException("The file " + fileName + " could not be opened.", true);
        return;
    }

    outFile << collums << " " << rows;

    for (int i = 0; i < collums; i++){
        outFile << "\n";
        for (int j = 0; j < rows; j++){
            outFile << matrix[i][j] << " ";
        }
    }

    outFile.close();
    cout << "success";
}