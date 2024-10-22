#include <iostream>
#include "NoteTransformer.h"
#include "FileUtils.h"
#include "Exception.h"
#include "Calculator.h"

using namespace std;

int main(){
    string modelPath = "THE MODEL FOLDER";
    string dataPath = "THE DATA FOLDER";
    cout << "NOTE TRANSFORMER TRAINING" << "\n \n";
    system("pause");
    ntParams parameters;
    parameters.context = 128;
    parameters.layerCount = 4;
    parameters.headsInLayers = 2;
    parameters.keyDims = 8;
    parameters.velocityDims = 8;
    parameters.prevDims = 2;
    parameters.nextDims = 2;
    parameters.absolutePosDims = 2;
    parameters.connectDims = 64;
    parameters.modelDims = 32;
    parameters.ffnDims = 64;
    NoteTransformer transformer(parameters);
    cout << "Transformer object created \n";
    string answer;
    cout << "Press enter to start training or enter special command \n";
    cin >> answer;
    if (answer.compare("{}!psodkladu_THIS_IS_CODE_FOR_RANDOM_INITIALIZATION_hfv{][bebe")){
        transformer.randomInit();
        transformer.save(modelPath);
    }
    try{
    transformer.init(modelPath);
    cout << "\nThe model has been initialized successfuly with " << Calculator::numberOfParameters(parameters) << " parameters \n";
    cout << "Start the training? \n";
    system ("pause");
    TrainingSettings settings = TrainingSettings();
    settings.setDataPath(dataPath);
    settings.setBatchSize(1);
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