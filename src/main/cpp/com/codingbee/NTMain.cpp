#include <iostream>
#include "NoteTransformer.h"
#include "FileUtils.h"
#include "Exception.h"

using namespace std;

int main(){
    cout << "NOTE TRANSFORMER TRAINING" << "\n \n";
    cout << "Randomly initialize the transformer?" << "\n";
    system("pause");
    NoteTransformer transformer(128, 4, 2, 16, 16, 2, 2, 2, 64, 32, 64);
    transformer.randomInit();
    try{
    transformer.save("C:/Users/theco/NoteTransformers/No1");
    transformer.init("C:/Users/theco/NoteTransformers/No1");
    cout << "The model has been initialized with " << transformer.getNumberOfParameters() << " parameters \n";
    cout << "Start the training? \n";
    system ("pause");
    TrainingSettings settings = TrainingSettings();
    settings.setDataPath("C:\\Users\\theco\\NoteTransformers\\Datasets\\haydn3\\");
    settings.setBatchSize(3);
    settings.setEpochs(1);
    settings.setLearningRate(0.002);
    settings.setSoftmaxTemperature(1.1);
    settings.setBeta_1(0.9);
    settings.setBeta_2(0.98);
    settings.setEpsilon(0.00000001);
    cout << "Settings are fine \n";
    transformer.train(settings);

    }catch (Exception e){
        cerr << e.getMessage();
    }catch(const std::exception& e){
        std::cerr << e.what() << '\n';
    }
    
    cout << "Training process has been finished." << "\n";
    cout << "Want to close this dialog and dealocate memory?" << "\n";
    system("pause");
    transformer.~NoteTransformer();
    return 0;
}