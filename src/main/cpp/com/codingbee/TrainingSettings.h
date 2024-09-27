#include <string>
#ifndef TRAININGSETTINGS_H
#define TRAININGSETTINGS_H

using namespace std;

class TrainingSettings{
    private:
    int epochs;
    float learningRate;
    float softmaxTemperature;
    string dataPath;
    int batchSize;

    public:
    int getEpochs(){return epochs;}
    void setEpochs(int n);

    float getLearningRate(){return learningRate;}
    void setLearningRate(float aplha);
    
    float getSoftmaxTemperature(){return softmaxTemperature;}
    void setSoftmaxTemperature(float temperature);

    string getDataPath(){return dataPath;}
    void setDataPath(string path);

    int getBatchSize(){return batchSize;}
    void setBatchSize(int n);
};

#endif