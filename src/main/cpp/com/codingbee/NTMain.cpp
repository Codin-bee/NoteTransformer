#include <iostream>
#include "NoteTransformer.h"
#include "FileUtils.h"

using namespace std;

void loadParameters(NoteTransformer transformer){
    string path = "C:/Users/theco/NoteTransformers/No1";
    transformer.randomInit();
    transformer.save(path);
    transformer.init(path);
    
}

void train(NoteTransformer transformer){
    int n = FileUtils::getNumberOfFilesInDir("paath");
}


int main(){
    cout << "NOTE TRANSFORMER TRAINING" << "\n";
    cout << "Start the training?" << "\n";
    system("pause");
    NoteTransformer transformer(128, 4, 2, 16, 16, 2, 2, 2, 64, 32, 64);
    //loadParameters(transformer);
    //train(transformer);
    cout << "Training process has been finished." << "\n";
    cout << "Want to close this dialog?" << "\n";
    system("pause");
    return 0;
}