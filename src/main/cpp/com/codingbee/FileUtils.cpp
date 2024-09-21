#include "FileUtils.h"
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