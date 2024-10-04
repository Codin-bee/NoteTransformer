#include "TrainingSettings.h"
#include "Exception.h"

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