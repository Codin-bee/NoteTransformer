#ifndef NOTE_TRANSFORMER_H
#define NOTE_TRANSFORMER_H

#include <thread>
#include <vector>
#include <math.h>
#include <string>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <dirent.h>

using namespace std;

class NoteTransformer
{
private:
    //Basic sizes
    int contextSize, layers, headsPerLayer;

    //Dimensions
    int d_keyEmbedding, d_velocityEmbedding, d_prevNoteEmbedding, d_nextNoteEmbedding, d_absolutePosition, d_connectingLayer, d_model, d_ffn;

    //Key and velocity ranges, determined by the midi file structure, should never change
    const int keyRange = 128;
    const int velocityRange = 128;

    //Embedding matricies
    int** keyEmbeddingMatrix;
    int** velocityEmbeddingMatrix;

    //Embedding alphas
    float* prevNoteAlphas;
    float* nextNoteAlphas;
    float* absolutePosAlphas;

    //Connecting layer
    float*** connectingLayerWeights;
    float* connectingLayerBiases;

    //FNN weights and biases
    float***** ffnWeights;
    float*** ffnBiases;

    //Attention matricies
    float**** quarryMatricies;
    float**** keyMatricies;
    float**** valueUpMatricies;
    float**** valueDownMatricies;

    //Unembedding
    float** unembeddingMatrix;

    //Other essentials and utilities
    int softmaxTemperature;
    int d_embedding;
    int d_attention;
    int sqrtD_k;
    int outputMatrixRows;

public:

    /// @brief Note Transformer Constructor, creates new instance of Note Transformer based on given parameters
    /// @param context context size of the model
    /// @param layerCount number of layers in the model
    /// @param headsInLayer number of heads in each attention block
    /// @param keyDims embedding dimensions used for the key value
    /// @param velocityDims embedding dimensions used for the key value
    /// @param prevDims embedding dimesnsions for the previous note
    /// @param nextDims embedding dimensions for the next note
    /// @param absolutePosDims embedding dimensions for the absolute position of the note
    /// @param connectDims dimensions of the hidden layer in the connecting FFN
    /// @param modelDims dimensions of the model
    /// @param ffnDims dimensions of the hidden layer in the feed forward networks
    NoteTransformer(int context, int layerCount, int headsInLayers, int keyDims, int velocityDims, int prevDims, 
    int nextDims, int absolutePosDims, int connectDims, int modelDims, int ffnDims){
            //Basic sizes
            contextSize = context;
            layers = layerCount;
            headsPerLayer = headsInLayers;
            //Dimensions
            d_keyEmbedding = keyDims;
            d_velocityEmbedding = velocityDims;
            d_prevNoteEmbedding = prevDims;
            d_nextNoteEmbedding = nextDims;
            d_absolutePosition = absolutePosDims;
            d_connectingLayer = connectDims;
            d_model = modelDims;
            d_ffn = ffnDims;
            //Utilities
            d_embedding = d_keyEmbedding + d_velocityEmbedding + d_prevNoteEmbedding + d_nextNoteEmbedding + d_absolutePosition;
            d_attention = d_model / headsPerLayer;
            sqrtD_k = sqrt(d_attention);
            softmaxTemperature = 1;
            outputMatrixRows = keyRange + velocityRange + 3;
    }

    void init(string dirPath){
        allocateModelMemory();
        /*TODO: implement initialization from directory*/
    }

    void randomInit(){
        allocateModelMemory();
         /*TODO: Implement initialization using random generated values*/
    }

    void save(string dirPath){
        /*TODO: implement saving matrix into given directory*/
    }
    
    /// @brief Frees all memory used by the Note Transformer
    ~NoteTransformer(){
        delete[] prevNoteAlphas;
        delete[] nextNoteAlphas;
        delete[] absolutePosAlphas;
        delete[] connectingLayerBiases;
        
        int i, j, k;
        for (i = 0; i < contextSize; i++)
        {
            delete[] keyEmbeddingMatrix[i];
            delete[] velocityEmbeddingMatrix[i];
        }
        
        for (i = 0; i < d_connectingLayer; i++)
        {
            delete[] connectingLayerWeights[0][i];
        }
        delete[] connectingLayerWeights[0];

        for (i = 0; i < d_model; i++)
        {
            delete[] connectingLayerWeights[1][i];
        }
        delete[] connectingLayerWeights[1];

        for (i = 0; i < layers; i++)
        {
            for (j = 0; j < contextSize; j++){
                delete[] ffnBiases[i][j];

                for (k = 0; k < d_ffn; k++){
                    delete[] ffnWeights[i][j][0][k];
                }

                for (k = 0; k < d_model; k++){
                    delete[] ffnWeights[i][j][1][k];
                }

                delete[] ffnWeights[i][j][0];
                delete[] ffnWeights[i][j][1];
                delete[] ffnWeights[i][j];
            }
            delete[] ffnWeights[i];

            for (j = 0; j < headsPerLayer; j++){
                for (k =0; k < d_model; k++){
                    delete[] quarryMatricies[i][j][k];
                    delete[] keyMatricies[i][j][k];
                    delete[] valueDownMatricies[i][j][k];
                }

                for (k =0; k < d_attention; k++){
                    delete[] valueUpMatricies[i][j][k];
                }

                quarryMatricies[i][j];
                keyMatricies[i][j];
                valueUpMatricies[i][j];
                valueDownMatricies[i][j];
            }

            quarryMatricies[i];
            keyMatricies[i];
            valueUpMatricies[i];
            valueDownMatricies[i];
            
        }

        for (i = 0; i < d_model; i++){
            delete[] unembeddingMatrix[i];
        }

        delete[] ffnBiases;
        delete[] ffnWeights;
        delete[] connectingLayerWeights;
        delete[] keyEmbeddingMatrix;
        delete[] velocityEmbeddingMatrix;
        delete[] quarryMatricies;
        delete[] keyMatricies;
        delete[] valueUpMatricies;
        delete[] valueDownMatricies;
        delete[] unembeddingMatrix;
    }


    /// @brief Processes given matrix
    /// @param matrixToProcess matrix containing the note values. First index is the note number, second
    ///                        is: 0-key; 1-velocity; 2-distance to previous note; 3-distance to next note; 4-absolute position of the note
    /// @return matrix containing predicted next notes. First index is note number, second is: 0-127 probabiites for key value; 128-255 
    ///         probabilities for velocity value; 256-258 timings (previous note, next note, absolute position)
    float** process(int** matrixToProcess){
        int i, j, k;
        //Embedding

        float** embeddedMatrix = new float*[contextSize];

        for (i = 0; i < contextSize; i++){

            embeddedMatrix[i] = new float[d_embedding];

                for (j = 0; j < d_keyEmbedding; j++){
                    embeddedMatrix[i][j] = keyEmbeddingMatrix[matrixToProcess[i][0]][j];
                }

                for (j = 0; j < d_velocityEmbedding; j++){
                    embeddedMatrix[i][j + d_keyEmbedding ] = velocityEmbeddingMatrix[matrixToProcess[i][1]][j];
                }

                for (j = 0; j < d_prevNoteEmbedding; j++){
                    embeddedMatrix[i][j + d_keyEmbedding + d_velocityEmbedding] = matrixToProcess[i][2] * prevNoteAlphas[j];
                }

                for (j = 0; j < d_nextNoteEmbedding; j++){
                    embeddedMatrix[i][j + d_keyEmbedding + d_velocityEmbedding + d_prevNoteEmbedding] = matrixToProcess[i][3] * nextNoteAlphas[j];
                }

                for (j = 0; j < d_absolutePosition; j++){
                    embeddedMatrix[i][j + d_keyEmbedding + d_velocityEmbedding + d_prevNoteEmbedding + d_nextNoteEmbedding] = matrixToProcess[i][4] * absolutePosAlphas[j];
                }
        }

                            //*Embeddings are stored inside embeddedMatrix

        //Connecting layer
        float** processedMatrix = new float*[contextSize];
    
        vector<thread> threads;
        for (i = 0; i < contextSize; i++){
            threads.push_back(
            std::thread([this, i, &embeddedMatrix, &processedMatrix]() {
                this->coonectLayer(embeddedMatrix[i], processedMatrix[i], i);
            }));

        }
       for (auto& t : threads){
            if (t.joinable()) {
                t.join();
            }
        }

                    //*Embeddings are stored in the processedMatrix
        
        for (int i = 0; i < layers; i++){
            int j;

            //Attention block
            float*** recivedChanges = new float**[headsPerLayer];
            for (j = 0; j < headsPerLayer; j++){
                threads.push_back(
                    std::thread(
                        [this, &processedMatrix, &recivedChanges, i, j]() {
                            this->attentionHead(processedMatrix, recivedChanges[j], i, j);
                        }
                    )
                );
                
            }

            for (auto& t : threads){
                if (t.joinable()){
                    t.join();
                }
            }

                    //*~~~The recivedChanges stores changes [head][token][d]

            for (j = 0; j < contextSize; j++){
                threads.push_back(
                    std::thread(
                        [this, &recivedChanges, &processedMatrix, j](){
                            this -> addChanges(processedMatrix[j], recivedChanges, j);
                        }
                    )
                );
            }

            for (auto& t : threads){
                if (t.joinable()){
                    t.join();
                }
            }
            
            //Feed forward networks
            for (j = 0; j < contextSize; j++)
            {
            threads.push_back(
                std::thread(
                    [this, i, j, &processedMatrix]() {
                    this->ffn(processedMatrix[j], i, j);
                    }
                )
            );
            }

            for (auto& t : threads){
                if (t.joinable()){
                    t.join();
                }
            }
            
        }
        //Unembedding
        float** finalOutput = new float*[contextSize];
        for (int i = 0; i < contextSize; i++){
            finalOutput[i] = new float[outputMatrixRows];
        }

        for (int i = 0; i < contextSize; i++){
            for (int j = 0; j < outputMatrixRows; j++){
                for (int k = 0; k < d_model; k++){
                    finalOutput[i][j] += processedMatrix[i][k] * unembeddingMatrix[j][k];
                }
            }
        }

        //Normalizing outputs
        for (i = 0; i < contextSize; i++){
            //Key probabilities
            float* tempArray = new float[keyRange];
            for (j = 0; j < keyRange; j++){
                tempArray[j] = finalOutput[i][j];
            }
            applySoftmax(tempArray, keyRange);

            for (j = 0; j < keyRange; j++){
                finalOutput[i][j] = tempArray[j];
            }
            delete[] tempArray;
            //Velocity probabilities
            tempArray = new float[velocityRange];
            for (j = keyRange; j < keyRange + velocityRange; j++){
                tempArray[j] = finalOutput[i][j];
            }
            applySoftmax(tempArray, keyRange);

            for (j = velocityRange; j < keyRange + velocityRange; j++){
                finalOutput[i][j] = tempArray[j];
            }

            //Timings (currently multiplication by thousand to make it easier for the model to get to numbers big enough to make sense)
            finalOutput[i][keyRange + velocityRange] = round(finalOutput[i][keyRange + velocityRange] * 1000);
            finalOutput[i][keyRange + velocityRange + 1] = round(finalOutput[i][keyRange + velocityRange + 1] * 1000);
            finalOutput[i][keyRange + velocityRange + 2] = round(finalOutput[i][keyRange + velocityRange + 2] * 1000);

        
        }
        //indexed[tokeNo][row] first 128 values : key probs|||second 128 values : velocity probs|||last 3 values : timings (prev, next, abs)
        return finalOutput;
    }

    void train(int iterations, int batchSize, string directoryPath){

        for (int i = 0; i < iterations; i++){

        }
    }

    private:

    void allocateModelMemory(){
        /*TODO: implement memory allocation for all network vectos, matricies and tensors*/
    }

    void coonectLayer(float* originalVector, float* downscaledVector, int vectorNo){
        downscaledVector = new float[d_model];

        float* upscaledVector = new float[d_connectingLayer];

        float neuronValue;

        int i, j;

        for (i = 0; i < d_connectingLayer; i++)
        {
            neuronValue = 0;
            for (j = 0; j < d_embedding; j++)
            {
               neuronValue += originalVector[j] * connectingLayerWeights[0][i][j];
            }
            upscaledVector[i] = neuronValue + connectingLayerBiases[i];
        }


        for (i = 0; i < d_model; i++)
        {
            neuronValue = 0;

            for (j = 0; j < d_connectingLayer; j++)
            {
                neuronValue += originalVector[j] * connectingLayerWeights[0][vectorNo][j];
            }
            downscaledVector[i] = neuronValue;
        }

        delete[] upscaledVector;
    }

    void applySoftmax(float* vector, int vectorLength){
        float sum = 0;
        int i;
        for (i = 0; i < vectorLength; i++)
        {
            sum += exp(vector[i] / softmaxTemperature);
        }
        for (i = 0; i < vectorLength; i++)
        {
            vector[i] = vector[i] / sum;
        }
    }

    void ffn(float* vector, int layer, int vectorNo){

        float* originalVector = new float[d_model];

        float* hiddenVector = new float[d_ffn];

        float neuronValue;

        int i, j;

        for (i = 0; i < d_ffn; i++)
        {
            originalVector[i] = vector[i];
            vector[i] = 0;
        }
        
        for (i = 0; i < d_ffn; i++){
            neuronValue = 0;
            for (j = 0; j < d_model; j++)
            {
                neuronValue += originalVector[i] * ffnWeights[layer][vectorNo][0][j][i];
            }
            hiddenVector[i] = neuronValue + ffnBiases[layer][vectorNo][i];
        }

        delete[] originalVector;

        for (i = 0; i < d_model; i++)
        {
            neuronValue = 0;
            for (j = 0; j < d_ffn; j++)
            {
                neuronValue += hiddenVector[i] * ffnWeights[layer][vectorNo][1][j][i];
            }
            vector[i] = neuronValue;
        }
        delete[] hiddenVector;
    }

    void attentionHead(float** theMatrix, float** outputMatrix, int layerNo, int headNo){
        //output[tokeNo][dimension(_model)]
        //Key, quarry and value calculation
        float** quarries = new float*[contextSize];
        float** keys = new float*[contextSize];
        float** values = new float*[contextSize];
        float** dotProducts = new float*[contextSize];

        for (int i = 0; i < contextSize; i++){
            quarries[i] = new float[d_attention];
            keys[i] = new float[d_attention];
            values[i] = new float[d_attention];
            dotProducts[i] = new float[contextSize];

            for (int j = 0; j < d_attention; j++){
                for (int k = 0; k < d_model; k++){
                    quarries[i][j] += quarryMatricies[layerNo][headNo][j][k] * theMatrix[i][k];
                    keys[i][j] += keyMatricies[layerNo][headNo][j][k] * theMatrix[i][k];
                    values[i][j] = valueDownMatricies[layerNo][headNo][j][k] * theMatrix[i][k];
                }
            }
        }

        //Key + quarry multiplication
        for (int i = 0; i < contextSize; i++){
            for (int j = 0; j < contextSize; j++){
                for (int k = 0; k < d_attention; k++){
                    dotProducts[i][j] += quarries[i][k] * keys[j][k];
                }
                dotProducts[i][j] /= sqrtD_k;
            }
        }

        //Masking
        for (int i = 0; i < contextSize; i++){
            for (int j = 0; j < contextSize; j++){
                if (i < j){
                    for (int k = 0; k < d_attention; k++){
                        dotProducts[i][j] = 0.000000001;
                    }
                }
            }
        }
        //Normalization
        for (int i = 0; i < contextSize; i++){
            applySoftmax(dotProducts[i], contextSize);
        }
        float** changes = new float*[contextSize];
        //Value multiplication
        for (int i = 0; i < contextSize; i++){
            changes[i] = new float[d_attention];
            for (int j = 0; j < contextSize; j++){
                for (int k = 0; k < d_attention; k++){
                    changes[i][k] += values[j][k] * dotProducts[i][j];
                }
            }
        }

        //Upscaling back to d_model
        outputMatrix = new float*[contextSize];
        for (int i = 0; i < contextSize; i++){
            outputMatrix[i] = new float[d_model];
            for (int j = 0; j < d_model; j++){
                for (int k = 0; k < d_attention; k++){
                    outputMatrix[i][j] += changes[i][j] * valueUpMatricies[layerNo][headNo][k][j];
                }
            }
        }
    }

    void addChanges(float* vector, float*** changes, int tokenNo){
        for (int i = 0; i < d_model; i++){
            for (int j = 0; j < headsPerLayer; j++){
                vector[i] += changes[j][tokenNo][i];
            }
        }
    }

    int getNumberOfFilesInDir(string directoryPath){
        DIR *dp;
  int i = 0;
  struct dirent *ep;     
  dp = opendir ("./");

  if (dp != NULL){
    while (ep = readdir (dp))
        i++;
        closedir (dp);
  }else{
    cerr << "Exception: the directory " + directoryPath + " could not been open.";
  }
  return i;
    }
    public :
    void saveMatrixToFile(const string& fileName, float** matrix, int collums, int rows){
        ofstream outFile(fileName);
        
        if (!outFile.is_open()){
            cerr << "Exception: the file " + fileName + "could not been open.";
            return;
        }

        outFile << collums << " " << rows;

        for (int i = 0; i < collums; i++){
            outFile << "\n";
            for (int j = 0; j < rows; j++){
                outFile << matrix[i][j] << " ";
            }
        }

        outFile.close();
        cout << "success";
    }
    private: 
    /// @brief multiplies two given matricies of the same size and than returns their product
    /// @param matrixA first matrix
    /// @param matrixB second matrix
    /// @param n size of the matricies
    /// @return the product of given matricies
    float** multiplySameSquareMatricies(float** matrixA, float** matrixB, int n){
        float ** matrixC = new float*[n];
        for (int i = 0; i < n; i++){
            matrixC[i] = new float[n];
            for (int k =0; k < n; k++){
                for (int j = 0; j < n; j++)
                {
                    matrixC[i][j] = matrixA[i][k] * matrixB[k][j];
                }
            }
        } 
    }

    //Getters and setters
    public :

    /// @brief returns the temperature(T) used for the softmax function
    /// @return the temperature
    float getSoftmaxTemperature(){
        return softmaxTemperature;
    }

    /// @brief Sets the new temperature(T) used in the softmax function
    /// @param t the new temperature
    void setSoftmaxTemperature(float t){
        softmaxTemperature = t;
    }
};
#endif