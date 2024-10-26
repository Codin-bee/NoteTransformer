#include <string>
#ifndef TRAININGSETTINGS_H
#define TRAININGSETTINGS_H

using namespace std;

class TrainingSettings{
    private:
    int epochs;
    float learningRate;
    float softmaxTemperature;
    float beta_1;
    float beta_2;
    float epsilon;
    string dataPath;
    string savePath;
    string adamParamsSavePath;
    bool loadOldAdamParams;
    int batchSize;

    public:
    TrainingSettings(string configPath);
    void saveTo(string configPath);

    int getEpochs(){return epochs;}
    void setEpochs(int n);

    float getLearningRate(){return learningRate;}
    void setLearningRate(float aplha);
    
    float getSoftmaxTemperature(){return softmaxTemperature;}
    void setSoftmaxTemperature(float temperature);

    float getBeta_1(){return beta_1;}
    void setBeta_1(float beta);

    float getBeta_2(){return beta_2;}
    void setBeta_2(float beta);
    
    float getEpsilon(){return epsilon;}
    void setEpsilon(float epsilon);

    string getDataPath(){return dataPath;}
    void setDataPath(string path);

    int getBatchSize(){return batchSize;}
    void setBatchSize(int n);

    string getSavePath(){return savePath;}
    void setSavePath(string path);

    string getAdamParamasSavePath(){return adamParamsSavePath;}
    void setAdamParamsSavePath(string path);

    bool doesLoadOldAdamParams(){return loadOldAdamParams;}
    void setLoadOldAdamParams(bool load);
};

#endif