#include "TrainingSettings.h"
#include "Exception.h"
#include "FileUtils.h"
#include <string>
#include <fstream>

TrainingSettings::TrainingSettings(string configPath){
    ifstream inFile(configPath + ".txt");
    if (!inFile.is_open()) {
        throw Exception("The file " + configPath + " could not been opened", ExceptionType::FILE_HANDLEING);
    }
    inFile >> epochs;
    inFile >> learningRate;
    inFile >> softmaxTemperature;
    inFile >> beta_1;
    inFile >> beta_2;
    inFile >> epsilon;
    inFile >> dataPath;
    inFile >> savePath;
    inFile >> adamParamsSavePath;
    inFile >> loadOldAdamParams;
    inFile >> batchSize;

    inFile.close();
}

TrainingSettings::TrainingSettings(){
    
}

void TrainingSettings::saveTo(string configPath){
    ofstream outFile(configPath + ".txt");
    if (!outFile.is_open()) {
        throw Exception("The file " + configPath + " could not been opened", ExceptionType::FILE_HANDLEING);
    }
    outFile << epochs << "\n";
    outFile << learningRate << "\n";
    outFile << softmaxTemperature << "\n";
    outFile << beta_1 << "\n";
    outFile << beta_2 << "\n";
    outFile << epsilon << "\n";
    outFile << dataPath << "\n";
    outFile << savePath << "\n";
    outFile << adamParamsSavePath << "\n";
    outFile << loadOldAdamParams << "\n";
    outFile << batchSize << "\n";

}

void TrainingSettings::setEpochs(int n){
    if (n <= 0){
        throw Exception("The number of epochs has to be bigger than 0", ExceptionType::INVALID_ARGUMENT);
    }
    epochs = n;
}

void TrainingSettings::setLearningRate(float rate){
    learningRate = rate;
}

void TrainingSettings::setSoftmaxTemperature(float temperature){
    if (temperature == 0){
        throw Exception("The softmax temperature is used for divission, so it can not be 0", ExceptionType::INVALID_ARGUMENT);
    }
    softmaxTemperature = temperature;
}

void TrainingSettings::setBeta_1(float beta){
    beta_1 = beta;
}

void TrainingSettings::setBeta_2(float beta){
    beta_2 = beta;
}

void TrainingSettings::setEpsilon(float eps){
    epsilon = eps;
}

void TrainingSettings::setDataPath(string path){
    if (path.empty()){
        throw Exception("The path to data can not be empty", ExceptionType::INVALID_ARGUMENT);
    }
    dataPath = path;
}

void TrainingSettings::setBatchSize(int n){
    if (n <= 0){
        throw Exception("The batch size has to be bigger than 0", ExceptionType::INVALID_ARGUMENT);
    }
    batchSize = n;
}

void TrainingSettings::setSavePath(string path){
    savePath = path;
}

void TrainingSettings::setAdamParamsSavePath(string path){
    adamParamsSavePath = path;
}

void TrainingSettings::setLoadOldAdamParams(bool load){
    loadOldAdamParams = load;
}