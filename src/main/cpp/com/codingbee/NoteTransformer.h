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

class NoteTransformer{
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
    int nextDims, int absolutePosDims, int connectDims, int modelDims, int ffnDims);

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
    /// @param matrixToProcess matrix containing the note values. First index is the note number, second is information
    ///                        type: 0-key; 1-velocity; 2-distance to previous note; 3-distance to next note; 4-absolute position of the note
    /// @return matrix containing predicted next notes. First index is note number, second is(at given indexes): 0-127 probabiites for key value; 128-255 
    ///         probabilities for velocity value; 256-258 timings (previous note, next note, absolute position)
    float** process(int** matrixToProcess);

    void train(int iterations, int batchSize, string directoryPath){

        for (int i = 0; i < iterations; i++){
            /*TODO: implement training algorithm*/
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