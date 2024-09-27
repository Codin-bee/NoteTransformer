#include "TrainingSettings.h"

void TrainingSettings::setEpochs(int n){
    epochs = n;
}

void TrainingSettings::setLearningRate(float rate){
    learningRate = rate;
}

void TrainingSettings::setSoftmaxTemperature(float temperature){
    softmaxTemperature = temperature;
}

void TrainingSettings::setDataPath(string path){
    dataPath = path;
}

void TrainingSettings::setBatchSize(int n){
    batchSize = n;
}