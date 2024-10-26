#include "NoteTransformer.h"
#include "FileUtils.h"
#include <random>
#include <filesystem>
#include "Exception.h"
#include "MemoryUtils.h"
#include <iostream>
#include "MathUtils.h"

using namespace std;

NoteTransformer::NoteTransformer(int context, int layerCount, int headsInLayers, int keyDims, int velocityDims, int prevDims, 
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
        attentionScalingFactor = 1 / sqrt(d_attention);
        softmaxTemperature = 1;
        outputMatrixColumns = keyRange + velocityRange + timingParams;
    }

NoteTransformer::NoteTransformer(ntParams parameters)
    : NoteTransformer(parameters.context, parameters.layerCount, parameters.headsInLayers, 
                      parameters.keyDims, parameters.velocityDims, parameters.prevDims, 
                      parameters.nextDims, parameters.absolutePosDims, parameters.connectDims, 
                      parameters.modelDims, parameters.ffnDims) {
}

float** NoteTransformer::process(int** matrixToProcess) {
    std::vector<std::thread> threads;
    // Embedding
    float** embeddedMatrix = embeddMatrix(matrixToProcess);
    // Connecting layer
    float** processedMatrix = new float*[contextSize];
    for (int i = 0; i < contextSize; i++) {
        processedMatrix[i] = new float[d_model];
        for (int j = 0; j < d_model; j++){
            processedMatrix[i][j] = 0;
        }
    }
    for (int i = 0; i < contextSize; i++) {
        threads.push_back(
            std::thread([this, i, &embeddedMatrix, &processedMatrix]() {
                this->connectLayer(embeddedMatrix[i], processedMatrix[i]);
            })
        );
    }
    joinAndClearThreads(threads);
    // Layers
    for (int i = 0; i < layers; i++) {
        // Attention block
        processAttention(processedMatrix, i);
        for (int j = 0; j < contextSize; j++){
            layerNormalizeVector(processedMatrix[j], i);
        }
        // Feed forward networks
        for (int j = 0; j < contextSize; j++) {
            threads.push_back(
                std::thread([this, i, j, &processedMatrix]() {
                    this->ffn(processedMatrix[j], i);
                })
            );
        }
        joinAndClearThreads(threads);
    }
    // Unembedding
    float** finalOutput = unembeddMatrixAndDeleteOriginal(processedMatrix);
    // Normalizing outputs
    normalizeOutputMatrix(finalOutput);
    return finalOutput;
}

void NoteTransformer::attentionHead(float** theMatrix, float**& outputMatrix, int layerNo, int headNo){
        //Key, quarry and value calculation
        float** quarries = MathUtils::multiplyMatricies(theMatrix, contextSize, d_model, quarryMatricies[layerNo][headNo], d_attention);
        float** keys =  MathUtils::multiplyMatricies(theMatrix, contextSize, d_model, keyMatricies[layerNo][headNo], d_attention);
        float** values = MathUtils::multiplyMatricies(theMatrix, contextSize, d_model, valueMatricies[layerNo][headNo], d_model);
        
        float** dotProducts = new float*[contextSize];
        for (int i = 0; i < contextSize; i++){
            dotProducts[i] = new float[contextSize];
            for (int j = 0; j < contextSize; j++){
                dotProducts[i][j] = 0;
            }
        }
        

        //Key + quarry multiplication
        for (int i = 0; i < contextSize; i++){
            for (int j = 0; j < contextSize; j++){
                float* products = MathUtils::multiplyVectors(quarries[i], keys[j], d_attention);
                dotProducts[i][j] = MathUtils::addVectorElements(products, d_attention);
                dotProducts[i][j] *= attentionScalingFactor;
                delete[] products;
            }
        }
        
        //Masking
        for (int i = 0; i < contextSize; i++){
            for (int j = 0; j < contextSize; j++){
                if (j < i){
                    //dotProducts[i][j] = - 300;
                }
            }
        }
        
        //Normalization
        for (int i = 0; i < contextSize; i++){
            MathUtils::applySoftmax(dotProducts[i], contextSize, softmaxTemperature);
        }
        //Calculating the changes to embeddings
        outputMatrix = MathUtils::multiplyMatricies(dotProducts, contextSize, contextSize, values, d_model);
        
        //Deleting used variables
        for (int i = 0; i < contextSize; i++){
            delete[] quarries[i];
            delete[] keys[i];
            delete[] values[i];
            delete[] dotProducts[i];
        }
    }

void NoteTransformer::addChanges(float*& vector, float*** changes, int tokenNo){
        for (int i = 0; i < d_model; i++){
            for (int j = 0; j < headsPerLayer; j++){
                vector[i] += changes[j][tokenNo][i];
            }
        }
    }

float NoteTransformer::getSoftmaxTemperature(){
        return softmaxTemperature;
    }

void NoteTransformer::setSoftmaxTemperature(float t){
        softmaxTemperature = t;
    }

void NoteTransformer::ffn(float*& vector, int layer){
        float* originalVector = new float[d_model];

        float* hiddenVector = new float[d_ffn];

        float neuronValue;

        int i, j;

        for (i = 0; i < d_model; i++){
            originalVector[i] = vector[i];
            vector[i] = 0;
        }
        for (i = 0; i < d_ffn; i++){
            neuronValue = 0;
            for (j = 0; j < d_model; j++){
                neuronValue += originalVector[j] * ffnWeights[layer][0][i][j];
            }
            hiddenVector[i] = MathUtils::leakyReLU(neuronValue + ffnBiases[layer][i]);
        }
        delete[] originalVector;
        for (i = 0; i < d_model; i++){
            neuronValue = 0;
            for (j = 0; j < d_ffn; j++){
                neuronValue += hiddenVector[j] * ffnWeights[layer][1][i][j];
            }
            vector[i] = MathUtils::leakyReLU(neuronValue);
        }
        delete[] hiddenVector;
    }

void NoteTransformer::connectLayer(float* originalVector, float*& downscaledVector){

        float* upscaledVector = new float[d_connectingLayer];

        float neuronValue;

        int i, j;

        for (i = 0; i < d_connectingLayer; i++){
            neuronValue = 0;
            for (j = 0; j < d_embedding; j++){
               neuronValue += originalVector[j] * connectingLayerWeights[0][i][j];
            }
            upscaledVector[i] = MathUtils::leakyReLU(neuronValue + connectingLayerBiases[i]);
        }


        for (i = 0; i < d_model; i++){
            neuronValue = 0;
            for (j = 0; j < d_connectingLayer; j++){
                neuronValue += upscaledVector[j] * connectingLayerWeights[1][i][j];
            }
            downscaledVector[i] = MathUtils::sigmoid(neuronValue);
        }

        delete[] upscaledVector;
    }

void NoteTransformer::allocateModelMemory(){
        //Embedding matricies
        MemoryUtils::allocateMatrix(keyEmbeddingMatrix, keyRange, d_keyEmbedding);
        MemoryUtils::allocateMatrix(velocityEmbeddingMatrix, velocityRange, d_velocityEmbedding);

        //Embedding aplhas
        prevNoteAlphas = new float[d_prevNoteEmbedding];
        nextNoteAlphas = new float[d_nextNoteEmbedding];
        absolutePosAlphas = new float[d_absolutePosition];

        //Connecting layer
        connectingLayerWeights = new float**[2];
        connectingLayerWeights[0] = new float*[d_connectingLayer];
        for (int i = 0; i < d_connectingLayer; i++){
                connectingLayerWeights[0][i] = new float[d_embedding];
        }
        connectingLayerWeights[1] = new float*[d_model];
        for (int i = 0; i < d_model; i++){
            connectingLayerWeights[1][i] = new float[d_connectingLayer];
        }
        connectingLayerBiases = new float[d_connectingLayer];

        //FFN weights and biases
        ffnWeights = new float***[layers];
        ffnBiases = new float*[layers];
        for (int i = 0; i < layers; i++){
            ffnWeights[i] = new float**[2];
            ffnWeights[i][0] = new float*[d_ffn];
            for (int j = 0; j < d_ffn; j++){
                ffnWeights[i][0][j] = new float[d_model];
            }
            ffnWeights[i][1] = new float*[d_model];
            for (int j = 0; j < d_model; j++){
                ffnWeights[i][1][j] =  new float[d_ffn];
            }
            ffnBiases[i] = new float[d_ffn];
        }

        //Attention matricies
        keyMatricies = new float ***[layers];
        quarryMatricies = new float ***[layers];
        valueMatricies = new float ***[layers];

        for (int i = 0; i < layers; i++){
            keyMatricies[i] = new float **[headsPerLayer];
            quarryMatricies[i] = new float **[headsPerLayer];
            valueMatricies[i] = new float **[headsPerLayer];
            for (int j = 0; j < headsPerLayer; j++){
            MemoryUtils::allocateMatrix(keyMatricies[i][j], d_model, d_attention);
            MemoryUtils::allocateMatrix(quarryMatricies[i][j], d_model, d_attention);
            MemoryUtils::allocateMatrix(valueMatricies[i][j], d_model, d_model);
            }
        }

        //Layer normalization
        MemoryUtils::allocateMatrix(betas, layers, d_model);
        MemoryUtils::allocateMatrix(gammas, layers, d_model);

        //Unembedding
        unembeddingMatrix = new float*[d_model];
        for (int i = 0; i < d_model; i++){
            unembeddingMatrix[i] = new float[outputMatrixColumns];
        }

    }

void NoteTransformer::train(TrainingSettings settings){
        int previousEpochs = 0;
        float g, m_hat, v_hat, time;
        float beta_1 = settings.getBeta_1(), beta_2 =  settings.getBeta_2(), beta_3 = 1 - beta_1, beta_4 = 1 - beta_2, alpha = settings.getLearningRate(), epsilon = settings.getEpsilon();
        //Allocation
        //Embedding matricies
        float** m_keyEmbedding;
        float** v_keyEmbedding;
        float** m_velocityEmbedding;
        float** v_velocityEmbedding;

        //Embedding aplhas
        float* m_prevNoteAlpha;
        float* v_prevNoteAlpha;
        float* m_nextNoteAlpha;
        float* v_nextNoteAlpha;
        float* m_absolutePos;
        float* v_absolutePos;

        //Connecting layer
        float*** m_connectingLayerWeights;
        float*** v_connectingLayerWeights;
        float* m_connectingLayerBiases;
        float* v_connectingLayerBiases;

        //FFN weights and biases
        float**** m_ffnWeights;
        float**** v_ffnWeights;
        float** m_ffnBiases;
        float** v_ffnBiases;

        //Attention matricies
        float**** m_keyMatricies;
        float**** v_keyMatricies;
        float**** m_quarryMatricies;
        float**** v_quarryMatricies;
        float**** m_valueMatricies;
        float**** v_valueMatricies;

        //Layer normalization
        float** m_betas;
        float** v_betas;
        float** m_gammas;
        float** v_gammas;

        //Unebedding
        float** m_unembeddingMatrix;
        float** v_unembeddingMatrix;
        if (settings.doesLoadOldAdamParams()){
            loadTrainingVariables(m_keyEmbedding, v_keyEmbedding, m_velocityEmbedding, v_velocityEmbedding, m_prevNoteAlpha, 
        v_prevNoteAlpha, m_nextNoteAlpha, v_nextNoteAlpha, m_absolutePos, v_absolutePos, m_connectingLayerWeights, v_connectingLayerWeights,
        m_connectingLayerBiases, v_connectingLayerBiases, m_ffnWeights, v_ffnWeights, m_ffnBiases, v_ffnBiases, m_keyMatricies, v_keyMatricies,
        m_quarryMatricies, v_quarryMatricies, m_valueMatricies, v_valueMatricies, m_betas, v_betas, m_gammas, v_gammas, m_unembeddingMatrix, 
        v_unembeddingMatrix, settings.getAdamParamasSavePath());
        previousEpochs = loadNumberOfPreviousEpochs(settings.getAdamParamasSavePath());
        }else{
            allocateTrainingVariables(m_keyEmbedding, v_keyEmbedding, m_velocityEmbedding, v_velocityEmbedding, m_prevNoteAlpha, 
        v_prevNoteAlpha, m_nextNoteAlpha, v_nextNoteAlpha, m_absolutePos, v_absolutePos, m_connectingLayerWeights, v_connectingLayerWeights,
        m_connectingLayerBiases, v_connectingLayerBiases, m_ffnWeights, v_ffnWeights, m_ffnBiases, v_ffnBiases, m_keyMatricies, v_keyMatricies,
        m_quarryMatricies, v_quarryMatricies, m_valueMatricies, v_valueMatricies, m_betas, v_betas, m_gammas, v_gammas, m_unembeddingMatrix, 
        v_unembeddingMatrix);
        }


        for (int epoch = previousEpochs; epoch < settings.getEpochs() + previousEpochs; epoch++){
            cout << "Training cost at the start of epoch " + to_string(epoch) + " : " << calculateAverageCost(settings.getDataPath(), 0, FileUtils::getNumberOfFilesInDir(settings.getDataPath()) / 2) << "\n";
            time = epoch + 1;
            for(int batchNo = 0; batchNo < FileUtils::getNumberOfFilesInDir(settings.getDataPath()) / settings.getBatchSize(); batchNo++){
                //Training data allocation
                int*** inputs = new int**[settings.getBatchSize()];
                float*** outputs = new float**[settings.getBatchSize()];
                for (int i = 0; i < settings.getBatchSize(); i++){
                    inputs[i] = FileUtils::readIntMatrixFromFile(settings.getDataPath() + "/input" + to_string(batchNo * settings.getBatchSize() + i));
                    outputs[i] = FileUtils::readFloatMatrixFromFile(settings.getDataPath() + "/output" + to_string(batchNo * settings.getBatchSize() + i));
                }
                save(settings.getSavePath());
                cout <<"Batch loaded \n";
                //Embedding matricies
                for (int i = 0; i < keyRange; i++){
                    for (int j = 0; j < d_keyEmbedding; j++){
                        g = calculateGradientWithRespectTo(keyEmbeddingMatrix[i], j, settings, inputs, outputs);
                        m_keyEmbedding[i][j] = beta_1 * m_keyEmbedding[i][j] + beta_3 * g;
                        v_keyEmbedding[i][j] = beta_2 * v_keyEmbedding[i][j] + beta_4 * pow(g, 2);
                        m_hat = m_keyEmbedding[i][j] / (1 - pow(beta_1, time));
                        v_hat = v_keyEmbedding[i][j] / (1 - pow(beta_2, time));
                        keyEmbeddingMatrix[i][j] = keyEmbeddingMatrix[i][j] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                    }
                }
                save(settings.getSavePath());
                cout << "Key embeddings \n";
                for (int i = 0; i < velocityRange; i++){
                    for (int j = 0; j < d_velocityEmbedding; j++){
                        g = calculateGradientWithRespectTo(velocityEmbeddingMatrix[i], j, settings, inputs, outputs);
                        m_velocityEmbedding[i][j] = beta_1 * m_velocityEmbedding[i][j] + beta_3 * g;
                        v_velocityEmbedding[i][j] = beta_2 * v_velocityEmbedding[i][j] + beta_4 * pow(g, 2);
                        m_hat = m_velocityEmbedding[i][j] / (1 - pow(beta_1, time));
                        v_hat = v_velocityEmbedding[i][j] / (1 - pow(beta_2, time));
                        velocityEmbeddingMatrix[i][j] = velocityEmbeddingMatrix[i][j] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                    }
                }
                save(settings.getSavePath());
                cout << "Velocity embeddings \n";

                //Embedding aplhas
                for (int i = 0; i < d_prevNoteEmbedding; i++){
                        g = calculateGradientWithRespectTo(prevNoteAlphas, i, settings, inputs, outputs);
                        m_prevNoteAlpha[i] = beta_1 * m_prevNoteAlpha[i] + beta_3 * g;
                        v_prevNoteAlpha[i] = beta_2 * v_prevNoteAlpha[i] + beta_4 * pow(g, 2);
                        m_hat = m_prevNoteAlpha[i] / (1 - pow(beta_1, time));
                        v_hat = v_prevNoteAlpha[i] / (1 - pow(beta_2, time));
                        prevNoteAlphas[i] = prevNoteAlphas[i] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                }
                save(settings.getSavePath());
                cout << "Previous note alphas \n";
                for (int i = 0; i < d_nextNoteEmbedding; i++){
                        g = calculateGradientWithRespectTo(nextNoteAlphas, i, settings, inputs, outputs);
                        m_nextNoteAlpha[i] = beta_1 * m_nextNoteAlpha[i] + beta_3 * g;
                        v_nextNoteAlpha[i] = beta_2 * v_nextNoteAlpha[i] + beta_4 * pow(g, 2);
                        m_hat = m_nextNoteAlpha[i] / (1 - pow(beta_1, time));
                        v_hat = v_nextNoteAlpha[i] / (1 - pow(beta_2, time));
                        nextNoteAlphas[i] = nextNoteAlphas[i] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                }
                save(settings.getSavePath());
                cout << "Next note aplhas \n";
                for (int i = 0; i < d_absolutePosition; i++){
                        g = calculateGradientWithRespectTo(absolutePosAlphas, i, settings, inputs, outputs);
                        m_absolutePos[i] = beta_1 * m_absolutePos[i] + beta_3 * g;
                        v_absolutePos[i] = beta_2 * v_absolutePos[i] + beta_4 * pow(g, 2);
                        m_hat = m_absolutePos[i] / (1 - pow(beta_1, time));
                        v_hat = v_absolutePos[i] / (1 - pow(beta_2, time));
                        absolutePosAlphas[i] = absolutePosAlphas[i] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                }
                save(settings.getSavePath());
                cout << "Absolute position alphas \n";

                //Connecting layer
                for (int i = 0; i < d_connectingLayer; i++){
                    for (int j = 0; j < d_embedding; j++){
                        g = calculateGradientWithRespectTo(connectingLayerWeights[0][i], j, settings, inputs, outputs);
                        m_connectingLayerWeights[0][i][j] = beta_1 * m_velocityEmbedding[i][j] + beta_3 * g;
                        v_connectingLayerWeights[0][i][j] = beta_2 * v_velocityEmbedding[i][j] + beta_4 * pow(g, 2);
                        m_hat = m_connectingLayerWeights[0][i][j] / (1 - pow(beta_1, time));
                        v_hat = v_connectingLayerWeights[0][i][j] / (1 - pow(beta_2, time));
                        connectingLayerWeights[0][i][j] = connectingLayerWeights[0][i][j] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                    }
                }
                for (int i = 0; i < d_model; i++){
                    for (int j = 0; j < d_connectingLayer; j++){
                        g = calculateGradientWithRespectTo(connectingLayerWeights[1][i], j, settings, inputs, outputs);
                        m_connectingLayerWeights[1][i][j] = beta_1 * m_velocityEmbedding[i][j] + beta_3 * g;
                        v_connectingLayerWeights[1][i][j] = beta_2 * v_velocityEmbedding[i][j] + beta_4 * pow(g, 2);
                        m_hat = m_connectingLayerWeights[1][i][j] / (1 - pow(beta_1, time));
                        v_hat = v_connectingLayerWeights[1][i][j] / (1 - pow(beta_2, time));
                        connectingLayerWeights[1][i][j] = connectingLayerWeights[i][0][j] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                    }
                }
                save(settings.getSavePath());
                cout << "Connecting layer weights\n";
                for (int i = 0; i < d_connectingLayer; i++){
                        g = calculateGradientWithRespectTo(connectingLayerBiases, i, settings, inputs, outputs);
                        m_connectingLayerBiases[i] = beta_1 * m_connectingLayerBiases[i] + beta_3 * g;
                        v_connectingLayerBiases[i] = beta_2 * v_connectingLayerBiases[i] + beta_4 * pow(g, 2);
                        m_hat = m_connectingLayerBiases[i] / (1 - pow(beta_1, time));
                        v_hat = v_connectingLayerBiases[i] / (1 - pow(beta_2, time));
                        connectingLayerBiases[i] = connectingLayerBiases[i] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                }
                save(settings.getSavePath());
                cout << "Connecting layer biases\n";

                //FFN weights and biases
                for (int i = 0; i < layers; i++){
                    for (int j = 0; j < d_ffn; j++){
                        for (int k = 0; k < d_model; k++){
                            g = calculateGradientWithRespectTo(ffnWeights[i][0][j], k, settings, inputs, outputs);
                            m_ffnWeights[i][0][j][k] = beta_1 * m_ffnWeights[i][0][j][k] + beta_3 * g;
                            v_ffnWeights[i][0][j][k] = beta_2 * v_ffnWeights[i][0][j][k] + beta_4 * pow(g, 2);
                            m_hat = m_ffnWeights[i][0][j][k] / (1 - pow(beta_1, time));
                            v_hat = v_ffnWeights[i][0][j][k] / (1 - pow(beta_2, time));
                            ffnWeights[i][0][j][k] = ffnWeights[i][0][j][k] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                        }
                    }
                    for (int j = 0; j < d_model; j++){
                        for (int k = 0; k < d_ffn; k++){
                            g = calculateGradientWithRespectTo(ffnWeights[i][1][j], k, settings, inputs, outputs);
                            m_ffnWeights[i][1][j][k] = beta_1 * m_ffnWeights[i][0][j][k] + beta_3 * g;
                            v_ffnWeights[i][1][j][k] = beta_2 * v_ffnWeights[i][0][j][k] + beta_4 * pow(g, 2);
                            m_hat = m_ffnWeights[i][1][j][k] / (1 - pow(beta_1, time));
                            v_hat = v_ffnWeights[i][1][j][k] / (1 - pow(beta_2, time));
                            ffnWeights[i][1][j][k] = ffnWeights[i][1][j][k] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                        }
                    }
                    for (int j = 0; j < d_ffn; j++){
                        g = calculateGradientWithRespectTo(ffnBiases[i], j, settings, inputs, outputs);
                        m_ffnBiases[i][j] = beta_1 * m_ffnBiases[i][j] + beta_3 * g;
                        v_ffnBiases[i][j] = beta_2 * v_ffnBiases[i][j] + beta_4 * pow(g, 2);
                        m_hat = m_ffnBiases[i][j] / (1 - pow(beta_1, time));
                        v_hat = v_ffnBiases[i][j] / (1 - pow(beta_2, time));
                        ffnBiases[i][j] = ffnBiases[i][j] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                    }
                }
                save(settings.getSavePath());
                cout << "FFNs \n";

                //Attention matricies
                for (int i = 0; i < layers; i++){
                    for (int j = 0; j < headsPerLayer; j++){
                        for (int k = 0; k < d_attention; k++){
                            for (int l = 0; l < d_model; l++){
                                g = calculateGradientWithRespectTo(keyMatricies[i][j][k], l, settings, inputs, outputs);
                                m_keyMatricies[i][j][k][l] = beta_1 * m_keyMatricies[i][j][k][l] + beta_3 * g;
                                v_keyMatricies[i][j][k][l] = beta_2 * v_keyMatricies[i][j][k][l] + beta_4 * pow(g, 2);
                                m_hat = m_keyMatricies[i][j][k][l] / (1 - pow(beta_1, time));
                                v_hat = v_keyMatricies[i][j][k][l] / (1 - pow(beta_2, time));
                                keyMatricies[i][j][k][l] = keyMatricies[i][j][k][l] - m_hat * (alpha / (sqrt(v_hat) + epsilon));

                                g = calculateGradientWithRespectTo(quarryMatricies[i][j][k], l, settings, inputs, outputs);
                                m_quarryMatricies[i][j][k][l] = beta_1 * m_quarryMatricies[i][j][k][l] + beta_3 * g;
                                v_quarryMatricies[i][j][k][l] = beta_2 * v_quarryMatricies[i][j][k][l] + beta_4 * pow(g, 2);
                                m_hat = m_quarryMatricies[i][j][k][l] / (1 - pow(beta_1, time));
                                v_hat = v_quarryMatricies[i][j][k][l] / (1 - pow(beta_2, time));
                                quarryMatricies[i][j][k][l] = quarryMatricies[i][j][k][l] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                                
                                g = calculateGradientWithRespectTo(valueMatricies[i][j][k], l, settings, inputs, outputs);
                                m_valueMatricies[i][j][k][l] = beta_1 * m_valueMatricies[i][j][k][l] + beta_3 * g;
                                v_valueMatricies[i][j][k][l] = beta_2 * v_valueMatricies[i][j][k][l] + beta_4 * pow(g, 2);
                                m_hat = m_valueMatricies[i][j][k][l] / (1 - pow(beta_1, time));
                                v_hat = v_valueMatricies[i][j][k][l] / (1 - pow(beta_2, time));
                                valueMatricies[i][j][k][l] = valueMatricies[i][j][k][l] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                            }
                        }
                    }
                }
                save(settings.getSavePath());
                cout << "Attention matricies \n";

                //Unembedding
                for (int i = 0; i < keyRange + velocityRange + 3; i++){
                    for (int j = 0; j < d_model; j++){
                        g = calculateGradientWithRespectTo(unembeddingMatrix[i], j, settings, inputs, outputs);
                        m_unembeddingMatrix[i][j] = beta_1 * m_unembeddingMatrix[i][j] + beta_3 * g;
                        v_unembeddingMatrix[i][j] = beta_2 * v_unembeddingMatrix[i][j] + beta_4 * pow(g, 2);
                        m_hat = m_unembeddingMatrix[i][j] / (1 - pow(beta_1, time));
                        v_hat = v_unembeddingMatrix[i][j] / (1 - pow(beta_2, time));
                        unembeddingMatrix[i][j] = unembeddingMatrix[i][j] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                    }
                }
                save(settings.getSavePath());
                cout << "Unembedding \n";

                //Layer normalization
                for (int i = 0; i < layers; i++){
                    for (int j = 0; j < d_model; j++){
                        g = calculateGradientWithRespectTo(gammas[i], j, settings, inputs, outputs);
                        m_gammas[i][j] = beta_1 * m_gammas[i][j] + beta_3 * g;
                        v_gammas[i][j] = beta_2 * v_gammas[i][j] + beta_4 * pow(g, 2);
                        m_hat = m_gammas[i][j] / (1 - pow(beta_1, time));
                        v_hat = v_gammas[i][j] / (1 - pow(beta_2, time));
                        gammas[i][j] = gammas[i][j] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                    }
                }
                for (int i = 0; i < layers; i++){
                    for (int j = 0; j < d_model; j++){
                        g = calculateGradientWithRespectTo(betas[i], j, settings, inputs, outputs);
                        m_betas[i][j] = beta_1 * m_betas[i][j] + beta_3 * g;
                        v_betas[i][j] = beta_2 * v_betas[i][j] + beta_4 * pow(g, 2);
                        m_hat = m_betas[i][j] / (1 - pow(beta_1, time));
                        v_hat = v_betas[i][j] / (1 - pow(beta_2, time));
                        betas[i][j] = betas[i][j] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                    }
                }
                save(settings.getSavePath());
                cout << "Layer normalization \n";

                //Training data deallocation
                for (int i = 0; i < settings.getBatchSize(); i++){
                for (int j = 0; j < contextSize; j++){
                    delete[] inputs[i][j];
                    delete[] outputs[i][j];
                }
                delete[] inputs[i];
                delete[] outputs[i];
            }
                delete[] inputs;
                delete[] outputs;

            }
        }
        cout << "Training cost at the end of training: " << calculateAverageCost(settings.getDataPath(), 0, FileUtils::getNumberOfFilesInDir(settings.getDataPath())) << "\n";
        //Saving Adam values
        saveTrainingVariables(m_keyEmbedding, v_keyEmbedding, m_velocityEmbedding, v_velocityEmbedding, m_prevNoteAlpha, 
        v_prevNoteAlpha, m_nextNoteAlpha, v_nextNoteAlpha, m_absolutePos, v_absolutePos, m_connectingLayerWeights, v_connectingLayerWeights,
        m_connectingLayerBiases, v_connectingLayerBiases, m_ffnWeights, v_ffnWeights, m_ffnBiases, v_ffnBiases, m_keyMatricies, v_keyMatricies,
        m_quarryMatricies, v_quarryMatricies, m_valueMatricies, v_valueMatricies, m_betas, v_betas, m_gammas, v_gammas, m_unembeddingMatrix, 
        v_unembeddingMatrix, settings.getAdamParamasSavePath());
    
        //Dealocation of Adam values
        deallocateTrainingVariables(m_keyEmbedding, v_keyEmbedding, m_velocityEmbedding, v_velocityEmbedding, m_prevNoteAlpha, 
        v_prevNoteAlpha, m_nextNoteAlpha, v_nextNoteAlpha, m_absolutePos, v_absolutePos, m_connectingLayerWeights, v_connectingLayerWeights,
        m_connectingLayerBiases, v_connectingLayerBiases, m_ffnWeights, v_ffnWeights, m_ffnBiases, v_ffnBiases, m_keyMatricies, v_keyMatricies,
        m_quarryMatricies, v_quarryMatricies, m_valueMatricies, v_valueMatricies, m_betas, v_betas, m_gammas, v_gammas, m_unembeddingMatrix, v_unembeddingMatrix);
    }

float** NoteTransformer::embeddMatrix(int** matrix){
    float** embeddedMatrix = new float*[contextSize];
    for (int i = 0; i < contextSize; i++) {
        embeddedMatrix[i] = new float[d_embedding];
        for (int j = 0; j < d_keyEmbedding; j++) {
            embeddedMatrix[i][j] = keyEmbeddingMatrix[matrix[i][0]][j];
        }

        for (int j = 0; j < d_velocityEmbedding; j++) {
            embeddedMatrix[i][j + d_keyEmbedding] = velocityEmbeddingMatrix[matrix[i][1]][j];
        }

        for (int j = 0; j < d_prevNoteEmbedding; j++) {
            embeddedMatrix[i][j + d_keyEmbedding + d_velocityEmbedding] = matrix[i][2] * prevNoteAlphas[j];
        }

        for (int j = 0; j < d_nextNoteEmbedding; j++) {
            embeddedMatrix[i][j + d_keyEmbedding + d_velocityEmbedding + d_prevNoteEmbedding] = matrix[i][3] * nextNoteAlphas[j];
        }

        for (int j = 0; j < d_absolutePosition; j++) {
            embeddedMatrix[i][j + d_keyEmbedding + d_velocityEmbedding + d_prevNoteEmbedding + d_nextNoteEmbedding] = matrix[i][4] * absolutePosAlphas[j];
        }
    }
    return embeddedMatrix;
}

float** NoteTransformer::unembeddMatrixAndDeleteOriginal(float**& matrix){
    float** finalOutput = new float*[contextSize];
    for (int j = 0; j < contextSize; j++) {
        finalOutput[j] = new float[outputMatrixColumns];
        for (int k = 0; k < outputMatrixColumns; k++) {
            finalOutput[j][k] = 0;
        }
    }
    finalOutput = MathUtils::multiplyMatricies(matrix, contextSize, d_model, unembeddingMatrix, keyRange + velocityRange + 3);
    for (int j = 0; j < contextSize; j++){
        delete[] matrix[j];
    }
    delete[] matrix;
    return finalOutput;
}

void NoteTransformer::joinAndClearThreads(vector<thread>& threads){
    for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
    threads.clear();
}

void NoteTransformer::normalizeOutputMatrix(float**& matrix){
    float* tempArray;
    for (int i = 0; i < contextSize; i++) {
        // Key probabilities
        tempArray = new float[keyRange];
        for (int j = 0; j < keyRange; j++) {
            tempArray[j] = matrix[i][j];
        }
        MathUtils::applySoftmax(tempArray, keyRange, softmaxTemperature);
        for (int j = 0; j < keyRange; j++) {
            matrix[i][j] = tempArray[j];
        }
        delete[] tempArray;

        // Velocity probabilities
        tempArray = new float[velocityRange];
        for (int j = keyRange; j < keyRange + velocityRange; j++) {
            tempArray[j - keyRange] = matrix[i][j];
        }
        MathUtils::applySoftmax(tempArray, velocityRange, softmaxTemperature);
        for (int j = keyRange; j < keyRange + velocityRange; j++) {
           matrix[i][j] = tempArray[j - keyRange];
        }
        delete[] tempArray;

        // Timings (currently multiplication by 100)
        matrix[i][keyRange + velocityRange] = round(matrix[i][keyRange + velocityRange] * 100);
        matrix[i][keyRange + velocityRange + 1] = round(matrix[i][keyRange + velocityRange + 1] * 100);
        matrix[i][keyRange + velocityRange + 2] = round(matrix[i][keyRange + velocityRange + 2] * 100);
    }
}

void NoteTransformer::processAttention(float**& matrix, int layer){
    float*** receivedChanges = new float**[headsPerLayer];
    vector<thread> threads;
    for (int i = 0; i < headsPerLayer; i++) {
        threads.push_back(
            std::thread([this, &matrix, &receivedChanges, layer, i]() {
                this->attentionHead(matrix, receivedChanges[i], layer, i);
            })
        );
    }
    joinAndClearThreads(threads);
    for (int i = 0; i < contextSize; i++) {
        threads.push_back(
            std::thread([this, &receivedChanges, &matrix, i]() {
                this->addChanges(matrix[i], receivedChanges, i);
            })
        );
    }
    joinAndClearThreads(threads);
    for (int i = 0; i < headsPerLayer; i++){
        for (int j = 0; j < contextSize; j++){
            delete[] receivedChanges[i][j];
        }
        delete[] receivedChanges[i];
    }
    delete[] receivedChanges;
}

void NoteTransformer::layerNormalizeVector(float*& vector, int layerNo){
    float* originalVector = new float[d_model];
    for (int i = 0; i < d_model; i++){
        originalVector[i] = vector[i];
    }
    double mean = 0;
    for (int i = 0; i < d_model; i++){
        mean += originalVector[i];
    }
    mean /= d_model;

    double variance = 0;
    for (int i = 0; i < d_model; i++){
        variance += pow((originalVector[i] - mean), 2);
    }
    variance /= d_model;

    for (int i = 0; i < d_model; i++){
        vector[i] = gammas[layerNo][i] * ((originalVector[i] - mean) / sqrt(variance + 0.00001)) + betas[layerNo][i];
    }
}

float NoteTransformer::calculateCost(int** input, float** expectedOutput){
    double cost  = 0;
    int j;
    float** recieved = process(input);
    for (int i = 0; i < contextSize; i++){
        for (j = 0; j < keyRange; j++){
            cost += pow((recieved[i][j] - expectedOutput[i][j]), 2) / 2;
        }
        for (j = keyRange; j < velocityRange; j++){
            cost += pow((recieved[i][j] - expectedOutput[i][j]), 2) / 2;
        }
        for (j = keyRange + velocityRange; j < keyRange + velocityRange + 3; j++){
            cost += abs((recieved[i][j] - expectedOutput[i][j])) * 0.001;
        }
    }
    for (j = 0; j < contextSize; j++){
        delete[] recieved[j];
    }
    delete[] recieved;
    return (float) cost;
}

float NoteTransformer::calculateGradientWithRespectTo(float*& array, int index, TrainingSettings settings, int startIndex, int endIndex){
    float nudge = 0.000001;
    float originalWeight = array[index];
    array[index] = originalWeight + nudge;
    float costHigher = calculateAverageCost(settings.getDataPath(), startIndex, endIndex);
    array[index] = originalWeight + nudge;
    float costLower = calculateAverageCost(settings.getDataPath(), startIndex, endIndex);
    array[index] = originalWeight;
    return (float) (costHigher - costLower) / (float) (2.0f * nudge);
}

float NoteTransformer::calculateGradientWithRespectTo(float*& array, int index, TrainingSettings settings, int*** inputs, float*** outputs){
    float nudge = 0.000001;
    float originalWeight = array[index];
    array[index] = originalWeight + nudge;
    float costHigher = calculateAverageCost(inputs, outputs, settings.getBatchSize());
    array[index] = originalWeight + nudge;
    float costLower = calculateAverageCost(inputs, outputs, settings.getBatchSize());
    array[index] = originalWeight;
    return (float) (costHigher - costLower) / (float) (2.0f * nudge);
}

float NoteTransformer::calculateAverageCost(string dirPath, int startIndex, int endIndex){
    float sum = 0;
    int n = 0;
    for (int i = startIndex; i < endIndex; i++){
        sum += calculateCost(FileUtils::readIntMatrixFromFile(dirPath + "/input" + to_string(i)), 
                FileUtils::readFloatMatrixFromFile(dirPath + "/output" + to_string(i)));
        n++;
    }
    return (float) sum / (float) n;
}

float NoteTransformer::calculateAverageCost(int*** inputs, float*** outputs, int batchSize){
    float sum = 0;
    int n = 0;
    for (int i = 0; i < batchSize; i++){
        sum += calculateCost(inputs[i], outputs[i]);
        n++;
    }
    return (float) sum / (float) n;
}

void NoteTransformer::save(string dirPath){
        int i, j;
        string currentPath = dirPath;
        try{
        std::filesystem::create_directories(dirPath);
        //Embedding matricies
        FileUtils::saveMatrixToFiles(dirPath + "/key_embedding", keyEmbeddingMatrix, keyRange, d_keyEmbedding);
        FileUtils::saveMatrixToFiles(dirPath + "/velocity_embedding", velocityEmbeddingMatrix, velocityRange, d_velocityEmbedding);
        //Embedding alphas
        FileUtils::saveVectorToFiles(dirPath + "/prev_note_alphas", prevNoteAlphas, d_prevNoteEmbedding);
        FileUtils::saveVectorToFiles(dirPath + "/next_note_alphas", nextNoteAlphas, d_nextNoteEmbedding);
        FileUtils::saveVectorToFiles(dirPath + "/abs_pos_alphas", absolutePosAlphas, d_absolutePosition);
        //Connecting layer
        string currentPath = dirPath + "/connecting_layer";
        std::filesystem::create_directory(currentPath);
        FileUtils::saveMatrixToFiles(currentPath + "/connection0", connectingLayerWeights[0], d_connectingLayer, d_embedding);
        FileUtils::saveMatrixToFiles(currentPath + "/connection1", connectingLayerWeights[1], d_model, d_connectingLayer);
        FileUtils::saveVectorToFiles(currentPath + "/biases", connectingLayerBiases, d_connectingLayer);
        //FFN weights and biases
        for (i = 0; i < layers; i++){
            currentPath = dirPath + "/layers/layer" + to_string(i) + "/ffn_weights";
            std::filesystem::create_directories(currentPath);
            FileUtils::saveMatrixToFiles(currentPath + "/connection0", ffnWeights[i][0], d_ffn, d_model);
            FileUtils::saveMatrixToFiles(currentPath + "/connection1", ffnWeights[i][1], d_model, d_ffn);
        }
        FileUtils::saveMatrixToFiles(dirPath + "/ffn_biases", ffnBiases, layers, d_ffn);
        //Attention matricies
        for (i = 0; i < layers; i++){
            for (j = 0; j < headsPerLayer; j++){
                currentPath = dirPath + "/layers/layer" + to_string(i) + "/attention/head" + to_string(j);
                std::filesystem::create_directories(currentPath);
                FileUtils::saveMatrixToFiles(currentPath + "/keyMatrix", keyMatricies[i][j], d_model, d_attention);
                FileUtils::saveMatrixToFiles(currentPath + "/quarryMatrix", quarryMatricies[i][j], d_model, d_attention);
                FileUtils::saveMatrixToFiles(currentPath + "/valueMatrix", valueMatricies[i][j], d_model, d_model);
            }
        }
        //Layer normalization
        FileUtils::saveMatrixToFiles(dirPath + "/betas", betas, layers, d_model);
        FileUtils::saveMatrixToFiles(dirPath + "/gammas", gammas, layers, d_model);
        //Unembedding
        FileUtils::saveMatrixToFiles(dirPath + "/unembedding", unembeddingMatrix, d_model, outputMatrixColumns);
        }catch (Exception e){
            cerr << e.getMessage() << "\n";
        }catch(const std::exception& e)
        {
            std::cerr << e.what() << '\n';
        }
        
    }

void NoteTransformer::randomInit(){
        allocateModelMemory();
        int i, j, k, l;

        std::default_random_engine generator;
        float variation;

        //Embeding matricies
        variation = sqrt(6.0f / ((float)keyRange * (float)d_keyEmbedding));
        std::normal_distribution<float> distribution1(-variation, variation);
        for (i = 0; i < keyRange; i++){
            for (j = 0; j < d_keyEmbedding; j++){
                keyEmbeddingMatrix[i][j] = distribution1(generator);
            }
        }
        variation = sqrt(6.0f / ((float)velocityRange * (float)d_velocityEmbedding));
        std::normal_distribution<float> distribution2(-variation, variation);
        for (i = 0; i < velocityRange; i++){
            for (j = 0; j < d_velocityEmbedding; j++){
                velocityEmbeddingMatrix[i][j] = distribution2(generator);
            }
        }

        //Embedding aplhas
        variation = sqrt((float)d_nextNoteEmbedding);
        std::normal_distribution<float> distribution3(-variation, variation);
        for (i = 0; i < d_prevNoteEmbedding; i++){
            prevNoteAlphas[i] = distribution3(generator);
        }
        variation = sqrt((float)d_prevNoteEmbedding);
        std::normal_distribution<float> distribution4(-variation, variation);
        for (i = 0; i < d_nextNoteEmbedding; i++){
            nextNoteAlphas[i] = distribution4(generator);
        }
        variation = sqrt((float)d_absolutePosition);
        std::normal_distribution<float> distribution5(-variation, variation);
        for (i = 0; i < d_absolutePosition; i++){
            absolutePosAlphas[i] = distribution5(generator);
        }

        //Connecting layer
        variation = sqrt(2.0f / pow((float)d_embedding, 2));
        std::normal_distribution<float> distribution6(0, variation);
        for (i = 0; i < d_connectingLayer; i++){
            for (j = 0; j < d_embedding; j++){
                connectingLayerWeights[0][i][j] = distribution6(generator);
            }
        }
        variation = sqrt(2.0f / pow((float)d_connectingLayer, 2));
        std::normal_distribution<float> distribution7(0, variation);
        for (i = 0; i < d_model; i++){
            for (j = 0; j < d_connectingLayer; j++){
                connectingLayerWeights[1][i][j] = distribution7(generator);
            }
        }

        for (i = 0; i < d_connectingLayer; i++){
            connectingLayerBiases[i] = 0;
        }

        //FFN weights and biases
        for (i = 0; i < layers; i++){
            variation = sqrt(2.0f / pow((float)d_model, 2));
            std::normal_distribution<float> distribution8(0, variation);
            for (j = 0; j < d_ffn; j++){
                for (k = 0; k < d_model; k++){
                    ffnWeights[i][0][j][k] = distribution8(generator);
                }
            }
            variation = sqrt(2.0f / pow((float)d_ffn, 2));
            std::normal_distribution<float> distribution9(0, variation);
            for (j = 0; j < d_model; j++){
                for (k = 0; k < d_ffn; k++){
                    ffnWeights[i][1][j][k] = distribution9(generator);
                }
            }
            for (j = 0; j < d_ffn; j++){
                ffnBiases[i][j] = 0;
            }
        }

        //Attention matricies
        variation = sqrt(2.0f / pow((float)d_model, 2));
        std::normal_distribution<float> distribution10(0, variation);
        for (i = 0; i < layers; i++){
            for (j = 0; j < headsPerLayer; j++){
                for (k = 0; k < d_model; k++){
                    for (l = 0; l < d_attention; l++){
                        keyMatricies[i][j][k][l] = distribution10(generator);
                        quarryMatricies[i][j][k][l] = distribution10(generator);
                    }
                    for (int l = 0; l < d_model; l++){
                        valueMatricies[i][j][k][l] = distribution10(generator);
                    }
                }
            }
        }

        //Layer normalization
        for (i = 0; i < layers; i++){
            for (j = 0; j < d_model; j++){
                betas[i][j] = 0;
                gammas[i][j] = 1;
            }
        }

        //Unembedding
        variation = sqrt(6.0f /(float) outputMatrixColumns);
        std::normal_distribution<float> distribution11(-variation, variation);
        for (i = 0; i < d_model; i++){
            for (j = 0; j < outputMatrixColumns; j++){
                unembeddingMatrix[i][j] = distribution11(generator);
            }
        }
    }

void NoteTransformer::init(string dirPath){
        allocateModelMemory();
        int i, j;
        string currentPath;

        //Embedding matricies
        keyEmbeddingMatrix = FileUtils::readFloatMatrixFromFile(dirPath + "/key_embedding");
        velocityEmbeddingMatrix = FileUtils::readFloatMatrixFromFile(dirPath + "/velocity_embedding");

        //Embedding alphas
        prevNoteAlphas = FileUtils::readFloatVectorFromFile(dirPath + "/prev_note_alphas");
        nextNoteAlphas = FileUtils::readFloatVectorFromFile(dirPath + "/next_note_alphas");
        absolutePosAlphas = FileUtils::readFloatVectorFromFile(dirPath + "/abs_pos_alphas");

        //Connecting layer
        currentPath = dirPath + "/connecting_layer";
        connectingLayerWeights[0] = FileUtils::readFloatMatrixFromFile(currentPath + "/connection0");
        connectingLayerWeights[1] = FileUtils::readFloatMatrixFromFile(currentPath + "/connection1");
        connectingLayerBiases = FileUtils::readFloatVectorFromFile(currentPath + "/biases");

        //FFN weights and biases
        for (i = 0; i < layers; i++){
            currentPath = dirPath + "/layers/layer" + to_string(i) + "/ffn_weights";
            ffnWeights[i][0] = FileUtils::readFloatMatrixFromFile(currentPath + "/connection0");
            ffnWeights[i][1] = FileUtils::readFloatMatrixFromFile(currentPath + "/connection1");
        }
        ffnBiases = FileUtils::readFloatMatrixFromFile(dirPath + "/ffn_biases");

        //Attention matricies
        for (i = 0; i < layers; i++){
            for (j = 0; j < headsPerLayer; j++){
                currentPath = dirPath + "/layers/layer" + to_string(i) + "/attention/head" + to_string(j);
                keyMatricies[i][j] = FileUtils::readFloatMatrixFromFile(currentPath + "/keyMatrix");
                quarryMatricies[i][j] = FileUtils::readFloatMatrixFromFile(currentPath + "/quarryMatrix");
                valueMatricies[i][j] = FileUtils::readFloatMatrixFromFile(currentPath + "/valueMatrix");
            }
        }

        //Layer normalization
        betas = FileUtils::readFloatMatrixFromFile(dirPath + "/betas");
        gammas = FileUtils::readFloatMatrixFromFile(dirPath + "/gammas");

        //Unembedding
        unembeddingMatrix = FileUtils::readFloatMatrixFromFile(dirPath + "/unembedding");
    }

void NoteTransformer::deallocateTrainingVariables(float**& m_ke, float**& v_ke, float**& m_ve, float**& v_ve, float*& m_pna, 
float*& v_pna, float*& m_nna, float*& v_nna, float*& m_ap, float*& v_ap, float***& m_clw, float***& v_clw, float*& m_clb, 
float*& v_clb, float****& m_ffnw, float****& v_ffnw, float**& m_ffnb, float**& v_ffnb, float****& m_km, float****& v_km, 
float****& m_qm, float****& v_qm, float****& m_vm, float****& v_vm, float**& m_bet, float**& v_bet, float**& m_gam, float**& v_gam,
float**& m_unm, float**& v_unm){
    //Embedding matricies
    MemoryUtils::deallocateMatrix(m_ke, keyRange);
    MemoryUtils::deallocateMatrix(v_ke, keyRange);
    MemoryUtils::deallocateMatrix(m_ve, velocityRange);
    MemoryUtils::deallocateMatrix(v_ve, velocityRange);

    //Embedding alphas
    delete[] m_pna;
    delete[] v_pna;
    delete[] m_nna;
    delete[] v_nna;
    delete[] m_ap;
    delete[] v_ap;

    //Connecting layer
    MemoryUtils::deallocateMatrix(m_clw[0], d_connectingLayer);
    MemoryUtils::deallocateMatrix(v_clw[0], d_connectingLayer);
    MemoryUtils::deallocateMatrix(v_clw[1], d_model);
    MemoryUtils::deallocateMatrix(v_clw[1], d_model);
    delete[] m_clw;
    delete[] v_clw;
    delete[] m_clb;
    delete[] v_clb;

    //FFN weights and biases
    for (int i = 0; i < layers; i++) {
    MemoryUtils::deallocateMatrix(m_ffnw[i][0], d_ffn);
    MemoryUtils::deallocateMatrix(v_ffnw[i][0], d_ffn);
    MemoryUtils::deallocateMatrix(v_ffnw[i][1], d_model);
    MemoryUtils::deallocateMatrix(v_ffnw[i][1], d_model);
        delete[] m_ffnb[i];
        delete[] v_ffnb[i];
    }
    delete[] m_ffnw;
    delete[] v_ffnw;
    delete[] m_ffnb;
    delete[] v_ffnb;

    // Attention matricies
    MemoryUtils::deallocate4DTensor(m_km, layers, headsPerLayer, d_model);
    MemoryUtils::deallocate4DTensor(v_km, layers, headsPerLayer, d_model);
    MemoryUtils::deallocate4DTensor(m_qm, layers, headsPerLayer, d_model);
    MemoryUtils::deallocate4DTensor(v_qm, layers, headsPerLayer, d_model);
    MemoryUtils::deallocate4DTensor(m_vm, layers, headsPerLayer, d_model);
    MemoryUtils::deallocate4DTensor(v_vm, layers, headsPerLayer, d_model);

    //Layer normalization
    MemoryUtils::deallocateMatrix(m_bet, layers);
    MemoryUtils::deallocateMatrix(v_bet, layers);
    MemoryUtils::deallocateMatrix(m_gam, layers);
    MemoryUtils::deallocateMatrix(v_gam, layers);

    // Unembedding
    MemoryUtils::deallocateMatrix(m_unm, d_model);
    MemoryUtils::deallocateMatrix(v_unm, d_model);
}

void NoteTransformer::allocateTrainingVariables(float**& m_ke, float**& v_ke, float**& m_ve, float**& v_ve, float*& m_pna, 
float*& v_pna, float*& m_nna, float*& v_nna, float*& m_ap, float*& v_ap, float***& m_clw, float***& v_clw, float*& m_clb, 
float*& v_clb, float****& m_ffnw, float****& v_ffnw, float**& m_ffnb, float**& v_ffnb, float****& m_km, float****& v_km, 
float****& m_qm, float****& v_qm, float****& m_vm, float****& v_vm, float**& m_bet, float**& v_bet, float**& m_gam, float**& v_gam,
float**& m_unm, float**& v_unm){
    //Embedding matricies
    m_ke = MemoryUtils::allocateMatrixWithZeros(keyRange, d_keyEmbedding);
    v_ke = MemoryUtils::allocateMatrixWithZeros(keyRange, d_keyEmbedding);
    m_ve = MemoryUtils::allocateMatrixWithZeros(velocityRange, d_velocityEmbedding);
    v_ve = MemoryUtils::allocateMatrixWithZeros(velocityRange, d_velocityEmbedding);

    //Embedding aplhas
    m_pna = new float[d_prevNoteEmbedding]();
    v_pna= new float[d_prevNoteEmbedding]();
    m_nna = new float[d_nextNoteEmbedding]();
    v_nna = new float[d_nextNoteEmbedding]();
    m_ap = new float[d_absolutePosition]();
    v_ap = new float[d_absolutePosition]();

    //Connecting layer
    m_clw = new float**[2];
    v_clw = new float**[2];
    MemoryUtils::allocateMatrixWithZeros(m_clw[0], d_connectingLayer, d_embedding);
    MemoryUtils::allocateMatrixWithZeros(v_clw[0], d_connectingLayer, d_embedding);
    MemoryUtils::allocateMatrixWithZeros(m_clw[1], d_model, d_connectingLayer);
    MemoryUtils::allocateMatrixWithZeros(v_clw[1], d_model, d_connectingLayer);
    m_clb = new float[d_connectingLayer]();
    v_clb = new float[d_connectingLayer]();

    //FFN weights and biases
    m_ffnw = new float***[layers];
    v_ffnw = new float***[layers];
        for (int i = 0; i < layers; i++){
            m_ffnw[i] = new float**[2];
            v_ffnw[i] = new float**[2];
            MemoryUtils::allocateMatrixWithZeros(v_ffnw[i][0], d_ffn, d_model);
            MemoryUtils::allocateMatrixWithZeros(v_ffnw[i][0], d_ffn, d_model);
            MemoryUtils::allocateMatrixWithZeros(m_ffnw[i][1], d_model, d_ffn);
            MemoryUtils::allocateMatrixWithZeros(v_ffnw[i][1], d_model, d_ffn);
        }
    m_ffnb = MemoryUtils::allocateMatrixWithZeros(layers, d_ffn);
    v_ffnb = MemoryUtils::allocateMatrixWithZeros(layers, d_ffn);

    //Attention matricies
    m_km = new float***[layers];
    v_km = new float***[layers];
    m_qm = new float***[layers];
    v_qm = new float***[layers];
    m_vm = new float***[layers];
    v_vm = new float***[layers];
    for (int i = 0; i < layers; i++){
        m_km[i] = new float**[headsPerLayer];
        v_km[i] = new float**[headsPerLayer];
        m_qm[i] = new float**[headsPerLayer];
        v_qm[i] = new float**[headsPerLayer];
        m_vm[i] = new float**[headsPerLayer];
        v_vm[i] = new float**[headsPerLayer];
        for (int j = 0; j < headsPerLayer; j++){
            MemoryUtils::allocateMatrixWithZeros(m_km[i][j], d_model, d_attention);
            MemoryUtils::allocateMatrixWithZeros(v_km[i][j], d_model, d_attention);
            MemoryUtils::allocateMatrixWithZeros(m_qm[i][j], d_model, d_attention);
            MemoryUtils::allocateMatrixWithZeros(v_qm[i][j], d_model, d_attention);
            MemoryUtils::allocateMatrixWithZeros(m_vm[i][j], d_model, d_model);
            MemoryUtils::allocateMatrixWithZeros(v_vm[i][j], d_model, d_model);
        }
    }

    //Layer normalization
    m_bet = MemoryUtils::allocateMatrixWithZeros(layers, d_model);
    v_bet = MemoryUtils::allocateMatrixWithZeros(layers, d_model);
    m_gam = MemoryUtils::allocateMatrixWithZeros(layers, d_model);
    v_gam = MemoryUtils::allocateMatrixWithZeros(layers, d_model);

    //Unebedding
    m_unm = MemoryUtils::allocateMatrixWithZeros(d_model, outputMatrixColumns);
    v_unm = MemoryUtils::allocateMatrixWithZeros(d_model, outputMatrixColumns);

}

void NoteTransformer::saveTrainingVariables(float**& m_ke, float**& v_ke, float**& m_ve, float**& v_ve, float*& m_pna, 
float*& v_pna, float*& m_nna, float*& v_nna, float*& m_ap, float*& v_ap, float***& m_clw, float***& v_clw, float*& m_clb, 
float*& v_clb, float****& m_ffnw, float****& v_ffnw, float**& m_ffnb, float**& v_ffnb, float****& m_km, float****& v_km, 
float****& m_qm, float****& v_qm, float****& m_vm, float****& v_vm, float**& m_bet, float**& v_bet, float**& m_gam, float**& v_gam,
float**& m_unm, float**& v_unm, string path){
    std::filesystem::create_directories(path);
    //Embedding matricies
    FileUtils::saveMatrixToFiles(path + "m_keyEmbedding", m_ke, keyRange, d_keyEmbedding);
    FileUtils::saveMatrixToFiles(path + "v_keyEmbedding", v_ke, keyRange, d_keyEmbedding);
    FileUtils::saveMatrixToFiles(path + "m_velocityEmbedding", m_ve, velocityRange, d_velocityEmbedding);
    FileUtils::saveMatrixToFiles(path + "v_velocityEmbedding", v_ve, velocityRange, d_velocityEmbedding);

    //Embedding alphas
    FileUtils::saveVectorToFiles(path + "m_prevNoteAlpha", m_pna, d_prevNoteEmbedding);
    FileUtils::saveVectorToFiles(path + "v_prevNoteAlpha", v_pna, d_prevNoteEmbedding);
    FileUtils::saveVectorToFiles(path + "m_nextNoteAlpha", m_nna, d_nextNoteEmbedding);
    FileUtils::saveVectorToFiles(path + "v_nextNoteAlpha", v_nna, d_nextNoteEmbedding);
    FileUtils::saveVectorToFiles(path + "m_absolutePosition", m_ap, d_absolutePosition);
    FileUtils::saveVectorToFiles(path + "v_absolutePosition", v_ap, d_absolutePosition);

    //Connecting layer
    FileUtils::saveMatrixToFiles(path + "m_connectingLayerWeights0", m_clw[0], d_connectingLayer, d_embedding);
    FileUtils::saveMatrixToFiles(path + "v_connectingLayerWeights0", v_clw[0], d_connectingLayer, d_embedding);
    FileUtils::saveMatrixToFiles(path + "m_connectingLayerWeights1", v_clw[1], d_model, d_connectingLayer);
    FileUtils::saveMatrixToFiles(path + "v_connectingLayerWeights1", v_clw[1], d_model, d_connectingLayer);
    FileUtils::saveVectorToFiles(path + "m_connectingLayerBiases", m_clb, d_connectingLayer);
    FileUtils::saveVectorToFiles(path + "v_connectingLayerBiases", v_clb, d_connectingLayer);

    //FFN weights and biases
    for (int i = 0; i < layers; i++) {
        std::filesystem::create_directories(path + "ffn/" + to_string(i));
        FileUtils::saveMatrixToFiles(path + "ffn/" + to_string(i) + "/m_weights0", m_ffnw[i][0], d_ffn, d_model);
        FileUtils::saveMatrixToFiles(path + "ffn/" + to_string(i) + "/v_weights0", v_ffnw[i][0], d_ffn, d_model);
        FileUtils::saveMatrixToFiles(path + "ffn/" + to_string(i) + "/m_weights1", v_ffnw[i][1], d_model, d_ffn);
        FileUtils::saveMatrixToFiles(path + "ffn/" + to_string(i) + "/v_weights1", v_ffnw[i][1], d_model, d_ffn);
    }
    FileUtils::saveMatrixToFiles(path + "m_ffnBiases", m_ffnb, d_model, d_ffn);
    FileUtils::saveMatrixToFiles(path + "v_ffnBiases", v_ffnb, d_model, d_ffn);


    // Attention matricies
    for (int i = 0; i < layers; i++){
        for (int j = 0; j < headsPerLayer; j++){
            std::filesystem::create_directories(path + "attention/" + to_string(i) + "/" + to_string(j));
            FileUtils::saveMatrixToFiles(path + "attention/" + to_string(i) + "/" + to_string(j) + "m_keyMatricies", m_km[i][j], d_model, d_attention);
            FileUtils::saveMatrixToFiles(path + "attention/" + to_string(i) + "/" + to_string(j) + "v_keyMatricies", v_km[i][j], d_model, d_attention);
            FileUtils::saveMatrixToFiles(path + "attention/" + to_string(i) + "/" + to_string(j) + "m_quarryMatricies", m_qm[i][j], d_model, d_attention);
            FileUtils::saveMatrixToFiles(path + "attention/" + to_string(i) + "/" + to_string(j) + "v_quarryMatricies", v_qm[i][j], d_model, d_attention);
            FileUtils::saveMatrixToFiles(path + "attention/" + to_string(i) + "/" + to_string(j) + "m_valueMatricies", m_vm[i][j], d_model, d_model);
            FileUtils::saveMatrixToFiles(path + "attention/" + to_string(i) + "/" + to_string(j) + "v_valueMatricies", v_vm[i][j], d_model, d_model);
        }
    }

    //Layer normalization
    FileUtils::saveMatrixToFiles(path + "m_betas", m_bet, layers, d_model);
    FileUtils::saveMatrixToFiles(path + "v_betas", v_bet, layers, d_model);
    FileUtils::saveMatrixToFiles(path + "m_gammas", m_gam, layers, d_model);
    FileUtils::saveMatrixToFiles(path + "v_gammas", v_gam, layers, d_model);

    // Unembedding
    FileUtils::saveMatrixToFiles(path + "m_unembedding", m_unm, d_model, outputMatrixColumns);
    FileUtils::saveMatrixToFiles(path + "v_unembedding", v_unm, d_model, outputMatrixColumns);
}

void NoteTransformer::loadTrainingVariables(float**& m_ke, float**& v_ke, float**& m_ve, float**& v_ve, float*& m_pna, 
float*& v_pna, float*& m_nna, float*& v_nna, float*& m_ap, float*& v_ap, float***& m_clw, float***& v_clw, float*& m_clb, 
float*& v_clb, float****& m_ffnw, float****& v_ffnw, float**& m_ffnb, float**& v_ffnb, float****& m_km, float****& v_km, 
float****& m_qm, float****& v_qm, float****& m_vm, float****& v_vm, float**& m_bet, float**& v_bet, float**& m_gam, float**& v_gam,
float**& m_unm, float**& v_unm, string path){
    //Embedding matricies
    m_ke = FileUtils::readFloatMatrixFromFile(path + "/m_keyEmbedding");
    v_ke = FileUtils::readFloatMatrixFromFile(path + "/v_keyEmbedding");
    m_ve = FileUtils::readFloatMatrixFromFile(path + "/m_velocityEmbedding");
    v_ve = FileUtils::readFloatMatrixFromFile(path + "/v_velocityEmbedding");

    //Embedding alphas
    m_pna = FileUtils::readFloatVectorFromFile(path + "/m_prevNoteAlpha");
    v_pna = FileUtils::readFloatVectorFromFile(path + "/v_prevNoteAlpha");
    m_nna = FileUtils::readFloatVectorFromFile(path + "/m_nextNoteAlpha");
    v_nna = FileUtils::readFloatVectorFromFile(path + "/v_nextNoteAlpha");
    m_ap = FileUtils::readFloatVectorFromFile(path + "/m_absolutePosition");
    v_ap = FileUtils::readFloatVectorFromFile(path + "/v_absolutePosition");

    //Connecting layer

    m_clw[0] = FileUtils::readFloatMatrixFromFile(path + "m_connectingLayerWeights0");
    v_clw[0] = FileUtils::readFloatMatrixFromFile(path + "v_connectingLayerWeights0");
    m_clw[1] = FileUtils::readFloatMatrixFromFile(path + "m_connectingLayerWeights1");
    v_clw[1] = FileUtils::readFloatMatrixFromFile(path + "v_connectingLayerWeights1");
    m_clb = FileUtils::readFloatVectorFromFile(path + "m_connectingLayerBiases");
    v_clb = FileUtils::readFloatVectorFromFile(path + "v_connectingLayerBiases");

    //FFN weights and biases
    m_ffnw = new float***[layers];
    v_ffnw = new float***[layers];
    for (int i = 0; i < layers; i++) {
        m_ffnw[i] = new float**[2];
        v_ffnw[i] = new float**[2];
        m_ffnw[i][0] = FileUtils::readFloatMatrixFromFile(path + "ffn/" + to_string(i) + "/m_weights0");
        v_ffnw[i][0] = FileUtils::readFloatMatrixFromFile(path + "ffn/" + to_string(i) + "/v_weights0");
        m_ffnw[i][1] = FileUtils::readFloatMatrixFromFile(path + "ffn/" + to_string(i) + "/m_weights1");
        v_ffnw[i][1] = FileUtils::readFloatMatrixFromFile(path + "ffn/" + to_string(i) + "/v_weights1");
    }
    m_ffnb = FileUtils::readFloatMatrixFromFile(path + "m_ffnBiases");
    v_ffnb = FileUtils::readFloatMatrixFromFile(path + "v_ffnBiases");

    // Attention matricies
    m_km = new float***[layers];
    v_km = new float***[layers];
    m_qm = new float***[layers];
    v_qm = new float***[layers];
    m_vm = new float***[layers];
    v_vm = new float***[layers];
    for (int i = 0; i < layers; i++){
        m_km[i] = new float**[headsPerLayer];
        v_km[i] = new float**[headsPerLayer];
        m_qm[i] = new float**[headsPerLayer];
        v_qm[i] = new float**[headsPerLayer];
        m_vm[i] = new float**[headsPerLayer];
        v_vm[i] = new float**[headsPerLayer];
        for (int j = 0; j < headsPerLayer; j++){
            m_km[i][j] = FileUtils::readFloatMatrixFromFile(path + "attention/" + to_string(i) + "/" + to_string(j) + "m_keyMatricies");
            v_km[i][j] = FileUtils::readFloatMatrixFromFile(path + "attention/" + to_string(i) + "/" + to_string(j) + "v_keyMatricies");
            m_qm[i][j] = FileUtils::readFloatMatrixFromFile(path + "attention/" + to_string(i) + "/" + to_string(j) + "m_quarryMatricies");
            v_qm[i][j] = FileUtils::readFloatMatrixFromFile(path + "attention/" + to_string(i) + "/" + to_string(j) + "v_quarryMatricies");
            m_vm[i][j] = FileUtils::readFloatMatrixFromFile(path + "attention/" + to_string(i) + "/" + to_string(j) + "m_valueMatricies");
            v_vm[i][j] = FileUtils::readFloatMatrixFromFile(path + "attention/" + to_string(i) + "/" + to_string(j) + "v_valueMatricies");
        }
    }

    //Layer normalization
    m_bet = FileUtils::readFloatMatrixFromFile(path + "m_betas");
    v_bet = FileUtils::readFloatMatrixFromFile(path + "v_betas");
    m_gam = FileUtils::readFloatMatrixFromFile(path + "m_gammas");
    v_gam = FileUtils::readFloatMatrixFromFile(path + "v_gammas");

    // Unembedding
    m_unm = FileUtils::readFloatMatrixFromFile(path + "m_unembedding");
    v_unm = FileUtils::readFloatMatrixFromFile(path + "v_unembedding");
}

int NoteTransformer::loadNumberOfPreviousEpochs(string path){
    return FileUtils::readIntVectorFromFile(path + "/epochs")[0];
}

NoteTransformer::~NoteTransformer() {
    //Embedding matricies
    MemoryUtils::deallocateMatrix(keyEmbeddingMatrix, keyRange);
    MemoryUtils::deallocateMatrix(velocityEmbeddingMatrix, velocityRange);
    
    //Embedding alphas
    delete[] prevNoteAlphas;
    delete[] nextNoteAlphas;
    delete[] absolutePosAlphas;
    delete[] connectingLayerBiases;

    //Connecting layer
    MemoryUtils::deallocateMatrix(connectingLayerWeights[0], d_connectingLayer);
    MemoryUtils::deallocateMatrix(connectingLayerWeights[1], d_model);
    delete[] connectingLayerWeights;

    //FFN weights and biases
    for (int i = 0; i < layers; i++) {
    MemoryUtils::deallocateMatrix(ffnWeights[i][1], d_model);
    MemoryUtils::deallocateMatrix(ffnWeights[i][0], d_ffn);
        delete[] ffnBiases[i];
        delete[] ffnWeights[i];
    }
    delete[] ffnWeights;
    delete[] ffnBiases;

    //Attention matricies
    MemoryUtils::deallocate4DTensor(quarryMatricies, layers, headsPerLayer, d_model);
    MemoryUtils::deallocate4DTensor(keyMatricies, layers, headsPerLayer, d_model);
    MemoryUtils::deallocate4DTensor(valueMatricies, layers, headsPerLayer, d_model);

    //Layer normalization
    MemoryUtils::deallocateMatrix(betas, layers);
    MemoryUtils::deallocateMatrix(gammas, layers);

    //Unembedding
    MemoryUtils::deallocateMatrix(unembeddingMatrix, outputMatrixColumns);
}
