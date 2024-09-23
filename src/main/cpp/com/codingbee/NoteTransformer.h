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
#include "MathUtils.h"

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

    void init(string dirPath);

    void randomInit();

    void save(string dirPath);
    
    /// @brief Frees all memory used by the Note Transformer
    ~NoteTransformer();


    /// @brief Processes given matrix
    /// @param matrixToProcess matrix containing the note values. First index is the note number, second is information
    ///                        type: 0-key; 1-velocity; 2-distance to previous note; 3-distance to next note; 4-absolute position of the note
    /// @return matrix containing predicted next notes. First index is note number, second is(at given indexes): 0-127 probabiites for key value; 128-255 
    ///         probabilities for velocity value; 256-258 timings (previous note, next note, absolute position)
    float** process(int** matrixToProcess);

    void train(int iterations, int batchSize, string directoryPath);

    private:

    void allocateModelMemory();

    void connectLayer(float* originalVector, float* downscaledVector, int vectorNo);

    void ffn(float* vector, int layer, int vectorNo);

    /// @brief calculates changes which should be added to given embeddings in order to make them represent their actual meaning in the context
    /// @param theMatrix the embeddings
    /// @param outputMatrix matrix which the method writes all changes into
    /// @param layerNo index of the vector in the processed matrix
    /// @param headNo index of the attention matricies
    void attentionHead(float** theMatrix, float** outputMatrix, int layerNo, int headNo);

    /// @brief Adds the given changes proposed by the attention heads to the vector
    /// @param vector the vector, which the function writes into
    /// @param changes the changes proposed by the attention head
    /// @param tokeNo index of the vector in the processed matrix
    void addChanges(float* vector, float*** changes, int tokenNo);

    //Getters and setters
    public :

    /// @brief returns the temperature(T) used for the softmax function
    /// @return the temperature
    float getSoftmaxTemperature();

    /// @brief Sets the new temperature(T) used in the softmax function
    /// @param t the new temperature
    void setSoftmaxTemperature(float t);
};
#endif
