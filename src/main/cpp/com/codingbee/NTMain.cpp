#include <iostream>
#include "NoteTransformer.h"
#include "FileUtils.h"

using namespace std;

int main(){
    cout << "NOTE TRANSFORMER TRAINING" << "\n";
    cout << "Start the training?" << "\n";
    system("pause");

    NoteTransformer transformer(128, 4, 2, 16, 16, 2, 2, 2, 64, 32, 64);
    transformer.randomInit();
    transformer.save("C:/Users/theco/NoteTransformers/No1");
    cout << "The model has been initialized with " << transformer.getNumberOfParameters() << " parameters \n";
    cout << "Training process has been finished." << "\n";
    cout << "Want to close this dialog?" << "\n";
    system("pause");
    transformer.~NoteTransformer();
    system("pause");
    return 0;
}