#ifndef NOTE_TRANSFORMER_H
#define NOTE_TRANSFORMER_H

#include <thread>
#include <vector>
#include <string>
#include "TrainingSettings.h"

using namespace std;
struct ntParams{
    int context;
    int layerCount;
    int headsInLayers;
    int keyDims;
    int velocityDims;
    int prevDims;
    int nextDims;
    int absolutePosDims;
    int connectDims;
    int modelDims;
    int ffnDims;
    };
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
    float** keyEmbeddingMatrix;
    float** velocityEmbeddingMatrix;

    //Embedding alphas
    float* prevNoteAlphas;
    float* nextNoteAlphas;
    float* absolutePosAlphas;
    //Number of tracked timing informations, determined by the model architecture, should never chnage
    const int timingParams = 3;

    //Connecting layer
    float*** connectingLayerWeights;
    float* connectingLayerBiases;

    //FFN weights and biases
    float**** ffnWeights;
    float** ffnBiases;

    //Attention matricies
    float**** quarryMatricies;
    float**** keyMatricies;
    float**** valueMatricies;

    //Layer normalization
    float** betas;
    float** gammas;

    //Unembedding
    float** unembeddingMatrix;

    //Other essentials and utilities
    int softmaxTemperature;
    int d_embedding;
    int d_attention;
    float attentionScalingFactor;
    int outputMatrixColumns;

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

    NoteTransformer(ntParams parameters);

    void init(string dirPath);

    void deallocateTrainingVariables(float **&m_ke, float **&v_ke, float **&m_ve, float **&v_ve, float *&m_pna, float *&v_pna, 
    float *&m_nna, float *&v_nna, float *&m_ap, float *&v_ap, float ***&m_clw, float ***&v_clw, float *&m_clb, float *&v_clb, 
    float ****&m_ffnw, float ****&v_ffnw, float **&m_ffnb, float **&v_ffnb, float ****&m_km, float ****&v_km, float ****&m_qm, 
    float ****&v_qm, float ****&m_vm, float ****&v_vm, float **&m_bet, float **&v_bet, float **&m_gam, float **&v_gam, 
    float **&m_unm, float **&v_unm);

    void saveTrainingVariables(float **&m_ke, float **&v_ke, float **&m_ve, float **&v_ve, float *&m_pna, float *&v_pna, 
    float *&m_nna, float *&v_nna, float *&m_ap, float *&v_ap, float ***&m_clw, float ***&v_clw, float *&m_clb, float *&v_clb, 
    float ****&m_ffnw, float ****&v_ffnw, float **&m_ffnb, float **&v_ffnb, float ****&m_km, float ****&v_km, float ****&m_qm, 
    float ****&v_qm, float ****&m_vm, float ****&v_vm, float **&m_bet, float **&v_bet, float **&m_gam, float **&v_gam, 
    float **&m_unm, float **&v_unm, string path);

    void randomInit();

    void save(string dirPath);

    float calculateCost(int** input, float** expectedOutput);

    float calculateAverageCost(string dirPath, int startIndex, int endIndex);
    
    /// @brief Frees all memory used by the Note Transformer
    ~NoteTransformer();


    /// @brief Processes given matrix
    /// @param matrixToProcess matrix containing the note values. First index is the note number, second is information
    ///                        type: 0-key; 1-velocity; 2-distance to previous note; 3-distance to next note; 4-absolute position of the note
    /// @return matrix containing predicted next notes. First index is note number, second is(at given indexes): 0-127 probabiites for key value; 128-255 
    ///         probabilities for velocity value; 256-258 timings (previous note, next note, absolute position)
    float** process(int** matrixToProcess);

    void train(TrainingSettings settings);
    
    void allocateModelMemory();

    private:

    void connectLayer(float* originalVector, float*& downscaledVector);

    /// @brief processes given vector using feed forward network with two transformations: first one with bias, second one without it
    /// @param vector processed vector
    /// @param layer index of the transformer layer
    void ffn(float*& vector, int layer);

    /// @brief calculates changes which should be added to given embeddings in order to make them represent their actual meaning in the context
    /// @param theMatrix the embeddings
    /// @param outputMatrix matrix which the method writes all changes into
    /// @param layer index of the transformer layer
    /// @param headNo index of the attention matricies
    void attentionHead(float** theMatrix, float**& outputMatrix, int layer, int headNo);

    /// @brief Adds the given changes proposed by the attention heads to the vector
    /// @param vector the vector, which the function writes into
    /// @param changes the changes proposed by the attention head
    /// @param tokeNo index of the vector in the processed matrix
    void addChanges(float*& vector, float*** changes, int tokenNo);

    void layerNormalizeVector(float*& vector, int layerNo);

    float calculateGradientWithRespectTo(float*& array, int index, TrainingSettings settings, int startIndex, int endIndex);

    float calculateGradientWithRespectTo(float*& array, int index, TrainingSettings settings, int*** inputs, float*** outputs);

    float calculateAverageCost(int*** inputs, float*** outputs, int batchSize);

    float** embeddMatrix(int** matrix);

    float** unembeddMatrixAndDeleteOriginal(float**& matrix);

    void normalizeOutputMatrix(float**& matrix);

    void processAttention(float**& matrix, int layer);

    void joinAndClearThreads(std::vector<thread>& threads);

    //Getters and setters
    public :

    /// @brief returns the temperature(T) used for the softmax function
    /// @return the temperature
    float getSoftmaxTemperature();

    /// @brief Sets the new temperature(T) used in the softmax function
    /// @param t the new temperature
    void setSoftmaxTemperature(float t);

    int getContextSize(){return contextSize;}

};
#endif