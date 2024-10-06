#include "NoteTransformer.h"
#include "FileUtils.h"
#include <random>
#include <filesystem>
#include "Exception.h"

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
        sqrtD_k = sqrt(d_attention);
        softmaxTemperature = 1;
        outputMatrixRows = keyRange + velocityRange + 3;
    }

float** NoteTransformer::process(int** matrixToProcess){
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

        //Connecting layer
        float** processedMatrix = new float*[contextSize];
        vector<thread> threads;
        for (i = 0; i < contextSize; i++){
            threads.push_back(
            std::thread([this, i, &embeddedMatrix, &processedMatrix]() {
                this->connectLayer(embeddedMatrix[i], processedMatrix[i]);
            }));

        }
       for (auto& t : threads){
            if (t.joinable()) {
                t.join();
            }
        }

        
        //Layers
        for (int i = 0; i < layers; i++){

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
                    this->ffn(processedMatrix[j], i);
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
        for (i = 0; i < contextSize; i++){
            finalOutput[i] = new float[outputMatrixRows];
        }

        for (i = 0; i < contextSize; i++){
            for (j = 0; j < outputMatrixRows; j++){
                for (k = 0; k < d_model; k++){
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
            MathUtils::applySoftmax(tempArray, keyRange, softmaxTemperature);

            for (j = 0; j < keyRange; j++){
                finalOutput[i][j] = tempArray[j];
            }
            delete[] tempArray;

            //Velocity probabilities
            tempArray = new float[velocityRange];
            for (j = keyRange; j < keyRange + velocityRange; j++){
                tempArray[j] = finalOutput[i][j];
            }
            MathUtils::applySoftmax(tempArray, keyRange, softmaxTemperature);

            for (j = velocityRange; j < keyRange + velocityRange; j++){
                finalOutput[i][j] = tempArray[j];
            }

            //Timings (currently multiplication by thousand to make it easier for the model to get to numbers big enough to make sense)
            finalOutput[i][keyRange + velocityRange] = round(finalOutput[i][keyRange + velocityRange] * 1000);
            finalOutput[i][keyRange + velocityRange + 1] = round(finalOutput[i][keyRange + velocityRange + 1] * 1000);
            finalOutput[i][keyRange + velocityRange + 2] = round(finalOutput[i][keyRange + velocityRange + 2] * 1000);

        
        }
        return finalOutput;
}

void NoteTransformer::attentionHead(float** theMatrix, float** outputMatrix, int layerNo, int headNo){
        int i, j, k;

        /*output[tokeNo][dimension(_model)]*/


        //Key, quarry and value calculation
        float** quarries = new float*[contextSize];
        float** keys = new float*[contextSize];
        float** values = new float*[contextSize];
        float** dotProducts = new float*[contextSize];

        for (i = 0; i < contextSize; i++){
            quarries[i] = new float[d_attention];
            keys[i] = new float[d_attention];
            values[i] = new float[d_attention];
            dotProducts[i] = new float[contextSize];

            for (j = 0; j < d_attention; j++){
                for (k = 0; k < d_model; k++){
                    quarries[i][j] += quarryMatricies[layerNo][headNo][j][k] * theMatrix[i][k];
                    keys[i][j] += keyMatricies[layerNo][headNo][j][k] * theMatrix[i][k];
                    values[i][j] = valueDownMatricies[layerNo][headNo][j][k] * theMatrix[i][k];
                }
            }
        }

        //Key + quarry multiplication
        for (i = 0; i < contextSize; i++){
            for (j = 0; j < contextSize; j++){
                for (k = 0; k < d_attention; k++){
                    dotProducts[i][j] += quarries[i][k] * keys[j][k];
                }
                dotProducts[i][j] /= sqrtD_k;
            }
        }

        //Masking
        for (i = 0; i < contextSize; i++){
            for (j = 0; j < contextSize; j++){
                if (i < j){
                    for (k = 0; k < d_attention; k++){
                        dotProducts[i][j] = 0.000000001;
                    }
                }
            }
        }
        //Normalization
        for (i = 0; i < contextSize; i++){
            MathUtils::applySoftmax(dotProducts[i], contextSize, softmaxTemperature);
        }
        float** changes = new float*[contextSize];
        //Value multiplication
        for (i = 0; i < contextSize; i++){
            changes[i] = new float[d_attention];
            for (j = 0; j < contextSize; j++){
                for (k = 0; k < d_attention; k++){
                    changes[i][k] += values[j][k] * dotProducts[i][j];
                }
            }
        }

        //Upscaling back to d_model
        outputMatrix = new float*[contextSize];
        for (i = 0; i < contextSize; i++){
            outputMatrix[i] = new float[d_model];
            for (j = 0; j < d_model; j++){
                for (k = 0; k < d_attention; k++){
                    outputMatrix[i][j] += changes[i][j] * valueUpMatricies[layerNo][headNo][k][j];
                }
            }
        }
    }

void NoteTransformer::addChanges(float* vector, float*** changes, int tokenNo){
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

void NoteTransformer::ffn(float* vector, int layer){

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
                neuronValue += originalVector[i] * ffnWeights[layer][0][j][i];
            }
            hiddenVector[i] = MathUtils::leakyReLU(neuronValue + ffnBiases[layer][i]);
        }



        delete[] originalVector;

        for (i = 0; i < d_model; i++)
        {
            neuronValue = 0;
            for (j = 0; j < d_ffn; j++)
            {
                neuronValue += hiddenVector[i] * ffnWeights[layer][1][j][i];
            }
            vector[i] = neuronValue;
        }
        delete[] hiddenVector;
    }

void NoteTransformer::connectLayer(float* originalVector, float* downscaledVector){
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
                neuronValue += originalVector[j] * connectingLayerWeights[1][i][j];
            }
            downscaledVector[i] = neuronValue;
        }

        delete[] upscaledVector;
    }

void NoteTransformer::allocateModelMemory(){
        int i, j, k;
        //Embedding matricies
        keyEmbeddingMatrix = new float*[keyRange];
        for (i = 0; i < keyRange; i++){
            keyEmbeddingMatrix[i] = new float[d_keyEmbedding];
        }
        velocityEmbeddingMatrix = new float*[velocityRange];
        for (i= 0; i < velocityRange; i++){
            velocityEmbeddingMatrix[i] = new float[d_velocityEmbedding];
        }

        //Embedding aplhas
        prevNoteAlphas = new float[d_prevNoteEmbedding];
        nextNoteAlphas = new float[d_nextNoteEmbedding];
        absolutePosAlphas = new float[d_absolutePosition];

        //Connecting layer
        connectingLayerWeights = new float**[2];
        connectingLayerWeights[0] = new float*[d_connectingLayer];
        for (i = 0; i < d_connectingLayer; i++){
                connectingLayerWeights[0][i] = new float[d_embedding];
        }
        connectingLayerWeights[1] = new float*[d_model];
        for (i = 0; i < d_model; i++){
            connectingLayerWeights[1][i] = new float[d_connectingLayer];
        }
        connectingLayerBiases = new float[d_connectingLayer];

        //FFN weights and biases
        ffnWeights = new float***[layers];
        ffnBiases = new float*[layers];
        for (i = 0; i < layers; i++){
            ffnWeights[i] = new float**[2];
            ffnWeights[i][0] = new float*[d_ffn];
            for (j = 0; j < d_ffn; j++){
                ffnWeights[i][0][j] = new float[d_model];
            }
            ffnWeights[i][1] = new float*[d_model];
            for (j = 0; j < d_model; j++){
                ffnWeights[i][1][j] =  new float[d_ffn];
            }
            ffnBiases[i] = new float[d_ffn];
        }

        //Attention matricies
        keyMatricies = new float ***[layers];
        quarryMatricies = new float ***[layers];
        valueUpMatricies = new float ***[layers];
        valueDownMatricies = new float ***[layers];

        for (i = 0; i < layers; i++){
            keyMatricies[i] = new float **[headsPerLayer];
            quarryMatricies[i] = new float **[headsPerLayer];
            valueUpMatricies[i] = new float **[headsPerLayer];
            valueDownMatricies[i] = new float **[headsPerLayer];
            for (j = 0; j < headsPerLayer; j++){
                keyMatricies[i][j] = new float *[d_attention];
                quarryMatricies[i][j] = new float *[d_attention];
                valueUpMatricies[i][j] = new float *[d_attention];
                valueDownMatricies[i][j] = new float *[d_attention];
                for (k = 0; k < d_attention; k++){
                    keyMatricies[i][j][k] = new float [d_model];
                    quarryMatricies[i][j][k] = new float [d_model];
                    valueUpMatricies[i][j][k] = new float [d_model];
                    valueDownMatricies[i][j][k] = new float [d_model];
                }
            }
        }

        //Unembedding
        unembeddingMatrix = new float*[keyRange + velocityRange + 3];
        for (i = 0; i < keyRange + velocityRange + 3; i++){
            unembeddingMatrix[i] = new float[d_model];
        }

    }

void NoteTransformer::train(TrainingSettings settings){
        int i, j, k, l;
        float g, m_hat, v_hat, time;
        float beta_1 = settings.getBeta_1(), beta_2 =  settings.getBeta_2(), beta_3 = 1 - beta_1, beta_4 = 1 - beta_2, alpha = settings.getLearningRate(), epsilon = settings.getEpsilon();

#pragma region  Allocation
        //Embedding matricies
        float** m_keyEmbedding = new float*[keyRange];
        float** v_keyEmbedding = new float*[keyRange];
        for (i = 0; i < keyRange; i++){
            m_keyEmbedding[i] = new float[d_keyEmbedding];
            v_keyEmbedding[i] = new float[d_keyEmbedding];
            for (j = 0; j < d_keyEmbedding; j++){
                m_keyEmbedding[i][j] = 0;
                v_keyEmbedding[i][j] = 0;
            }
        }
        
        float** m_velocityEmbedding = new float*[velocityRange];
        float** v_velocityEmbedding = new float*[velocityRange];
        for (i = 0; i < keyRange; i++){
            m_velocityEmbedding[i] = new float[d_velocityEmbedding];
            v_velocityEmbedding[i] = new float[d_velocityEmbedding];
            for (j = 0; j < d_velocityEmbedding; j++){
                m_velocityEmbedding[i][j] = 0;
                v_velocityEmbedding[i][j] = 0;
            }
        }

        //Embedding aplhas
        float* m_prevNoteAlpha = new float[d_prevNoteEmbedding];
        float* v_prevNoteAlpha = new float[d_prevNoteEmbedding];
        for (i = 0; i < d_prevNoteEmbedding; i++){
            m_prevNoteAlpha[i] = 0;
            v_prevNoteAlpha[i] = 0;
        } 

        float* m_nextNoteAlpha = new float[d_nextNoteEmbedding];
        float* v_nextNoteAlpha = new float[d_nextNoteEmbedding];
        for (i = 0; i < d_nextNoteEmbedding; i++){
            m_nextNoteAlpha[i] = 0;
            v_nextNoteAlpha[i] = 0;
        } 

        float* m_absolutePos = new float[d_absolutePosition];
        float* v_absolutePos = new float[d_absolutePosition];
        for (i = 0; i < d_absolutePosition; i++){
            m_absolutePos[i] = 0;
            v_absolutePos[i] = 0;
        }

        //Connecting layer
        float*** m_connectingLayerWeights= new float**[2];
        float*** v_connectingLayerWeights = new float**[2];
        m_connectingLayerWeights[0]= new float*[d_connectingLayer];
        v_connectingLayerWeights[0] = new float*[d_connectingLayer];
        m_connectingLayerWeights[1]= new float*[d_model];
        v_connectingLayerWeights[1] = new float*[d_model];

        for (i = 0; i < d_connectingLayer; i++){
            m_connectingLayerWeights[0][i] = new float[d_embedding];
            v_connectingLayerWeights[0][i] = new float[d_embedding];
            for (j = 0; j < d_embedding; j++){
                m_connectingLayerWeights[0][i][j] = 0;
                v_connectingLayerWeights[0][i][j] = 0;
            }
        }
        for (i = 0; i < d_model; i++){
            m_connectingLayerWeights[1][i] = new float[d_connectingLayer];
            v_connectingLayerWeights[1][i] = new float[d_connectingLayer];
            for (j = 0; j < d_connectingLayer; j++){
                m_connectingLayerWeights[1][i][j] = 0;
                v_connectingLayerWeights[1][i][j] = 0;
            }
        }

        float* m_connectingLayerBiases = new float[d_connectingLayer];
        float* v_connectingLayerBiases = new float[d_connectingLayer];
        for (i = 0; i < d_connectingLayer; i++){
            m_connectingLayerBiases[i] = 0;
            v_connectingLayerBiases[i] = 0;
        }

        //FFN weights and biases
        float**** m_ffnWeights= new float***[layers];
        float**** v_ffnWeights = new float***[layers];
        float** m_ffnBiases = new float*[layers];
        float** v_ffnBiases = new float*[layers];
        for (i = 0; i < layers; i++){
            m_ffnWeights[i] = new float**[2];
            v_ffnWeights[i] = new float**[2];

            m_ffnWeights[i][0]= new float*[d_ffn];
            v_ffnWeights[i][0] = new float*[d_ffn];

            m_ffnWeights[i][1] = new float*[d_model];
            v_ffnWeights[i][1] = new float*[d_model];

            for (j = 0; j < d_ffn; j++){
                m_ffnWeights[i][0][j]= new float[d_model];
                v_ffnWeights[i][0][j] = new float[d_model];
                for (k = 0; k < d_model; k++){ 
                    m_ffnWeights[i][0][j][k]= 0;
                    v_ffnWeights[i][0][j][k] = 0;
                }
            }
            
            for (j = 0; j < d_model; j++){
                m_ffnWeights[i][1][j]= new float[d_ffn];
                v_ffnWeights[i][1][j] = new float[d_ffn];
                for (k = 0; k < d_ffn; k++){ 
                    m_ffnWeights[i][1][j][k]= 0;
                    v_ffnWeights[i][1][j][k] = 0;
                }
            }

            m_ffnBiases[i] = new float[d_ffn];
            v_ffnBiases[i] = new float[d_ffn];
            for (j = 0; j < d_ffn; j++){
                m_ffnBiases[i][j] = 0;
                v_ffnBiases[i][j] = 0;
            }
        }

        //Attention matricies
        float**** m_keyMatricies = new float***[layers];
        float**** v_keyMatricies = new float***[layers];
        float**** m_quarryMatricies = new float***[layers];
        float**** v_quarryMatricies = new float***[layers];
        float**** m_valueUpMatricies = new float***[layers];
        float**** v_valueUpMatricies = new float***[layers];
        float**** m_valueDownMatricies = new float***[layers];
        float**** v_valueDownMatricies = new float***[layers];
        for (i = 0; i < layers; i++){
            m_keyMatricies[i] = new float**[headsPerLayer];
            v_keyMatricies[i] = new float**[headsPerLayer];
            m_quarryMatricies[i] = new float**[headsPerLayer];
            v_quarryMatricies[i] = new float**[headsPerLayer];
            m_valueUpMatricies[i] = new float**[headsPerLayer];
            v_valueUpMatricies[i] = new float**[headsPerLayer];
            m_valueDownMatricies[i] = new float**[headsPerLayer];
            v_valueDownMatricies[i] = new float**[headsPerLayer];
            for (j = 0; j < headsPerLayer; j++){
                m_keyMatricies[i][j] = new float*[d_attention];
                v_keyMatricies[i][j] = new float*[d_attention];
                m_quarryMatricies[i][j] = new float*[d_attention];
                v_quarryMatricies[i][j] = new float*[d_attention];
                m_valueUpMatricies[i][j] = new float*[d_attention];
                v_valueUpMatricies[i][j] = new float*[d_attention];
                m_valueDownMatricies[i][j] = new float*[d_attention];
                v_valueDownMatricies[i][j] = new float*[d_attention];
                for (k = 0; k < d_attention; k++){
                    m_keyMatricies[i][j][k] = new float[d_model];
                    v_keyMatricies[i][j][k] = new float[d_model];
                    m_quarryMatricies[i][j][k] = new float[d_model];
                    v_quarryMatricies[i][j][k] = new float[d_model];
                    m_valueUpMatricies[i][j][k] = new float[d_model];
                    v_valueUpMatricies[i][j][k] = new float[d_model];
                    m_valueDownMatricies[i][j][k] = new float[d_model];
                    v_valueDownMatricies[i][j][k] = new float[d_model];
                    for (l = 0; l < d_model; l++){
                        m_keyMatricies[i][j][k][j] = 0;
                        v_keyMatricies[i][j][k][j] = 0;
                        m_quarryMatricies[i][j][k][j] = 0;
                        v_quarryMatricies[i][j][k][j] = 0;
                        m_valueUpMatricies[i][j][k][j] = 0;
                        v_valueUpMatricies[i][j][k][j] = 0;
                        m_valueDownMatricies[i][j][k][j] = 0;
                        v_valueDownMatricies[i][j][k][j] = 0;
                    }
                }
            }
        }
        //Unebedding
        float** m_unembeddingMatrix = new float*[keyRange + velocityRange + 3];
        float** v_unembeddingMatrix = new float*[keyRange + velocityRange + 3];
        for (i = 0; i < keyRange + velocityRange + 3; i++){
            m_unembeddingMatrix[i] = new float[d_model];
            v_unembeddingMatrix[i] = new float[d_model];
            for (j = 0; j < d_model; j++){
                m_unembeddingMatrix[i][j] = 0;
                v_unembeddingMatrix[i][j] = 0;
            }
        }

#pragma endregion

        for (int epoch = 0; epoch < settings.getEpochs(); epoch++){
            time = epoch + 1;
            for(int batchNo = 0; batchNo < FileUtils::getNumberOfFilesInDir(settings.getDataPath()) / settings.getBatchSize(); batchNo++){
                int startIndex = batchNo * settings.getBatchSize();
                int endIndex = startIndex + settings.getBatchSize();
                //Embedding matricies
                for (i = 0; i < keyRange; i++){
                    for (j = 0; j < d_keyEmbedding; j++){
                        g = calculateGradientWithRespectTo(keyEmbeddingMatrix[i], j, settings, startIndex, endIndex);
                        m_keyEmbedding[i][j] = beta_1 * m_keyEmbedding[i][j] + beta_3 * g;
                        v_keyEmbedding[i][j] = beta_2 * v_keyEmbedding[i][j] + beta_4 * pow(g, 2);
                        m_hat = m_keyEmbedding[i][j] / (1 - pow(beta_1, time));
                        v_hat = v_keyEmbedding[i][j] / (1 - pow(beta_2, time));
                        keyEmbeddingMatrix[i][j] = keyEmbeddingMatrix[i][j] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                    }
                }
                for (i = 0; i < velocityRange; i++){
                    for (j = 0; j < d_velocityEmbedding; j++){
                        g = calculateGradientWithRespectTo(velocityEmbeddingMatrix[i], j, settings, startIndex, endIndex);
                        m_velocityEmbedding[i][j] = beta_1 * m_velocityEmbedding[i][j] + beta_3 * g;
                        v_velocityEmbedding[i][j] = beta_2 * v_velocityEmbedding[i][j] + beta_4 * pow(g, 2);
                        m_hat = m_velocityEmbedding[i][j] / (1 - pow(beta_1, time));
                        v_hat = v_velocityEmbedding[i][j] / (1 - pow(beta_2, time));
                        velocityEmbeddingMatrix[i][j] = velocityEmbeddingMatrix[i][j] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                    }
                }

                //Embedding aplhas
                for (i = 0; i < d_prevNoteEmbedding; i++){
                        g = calculateGradientWithRespectTo(prevNoteAlphas, i, settings, startIndex, endIndex);
                        m_prevNoteAlpha[i] = beta_1 * m_prevNoteAlpha[i] + beta_3 * g;
                        v_prevNoteAlpha[i] = beta_2 * v_prevNoteAlpha[i] + beta_4 * pow(g, 2);
                        m_hat = m_prevNoteAlpha[i] / (1 - pow(beta_1, time));
                        v_hat = v_prevNoteAlpha[i] / (1 - pow(beta_2, time));
                        prevNoteAlphas[i] = prevNoteAlphas[i] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                }
                for (i = 0; i < d_nextNoteEmbedding; i++){
                        g = calculateGradientWithRespectTo(nextNoteAlphas, i, settings, startIndex, endIndex);
                        m_nextNoteAlpha[i] = beta_1 * m_nextNoteAlpha[i] + beta_3 * g;
                        v_nextNoteAlpha[i] = beta_2 * v_nextNoteAlpha[i] + beta_4 * pow(g, 2);
                        m_hat = m_nextNoteAlpha[i] / (1 - pow(beta_1, time));
                        v_hat = v_nextNoteAlpha[i] / (1 - pow(beta_2, time));
                        nextNoteAlphas[i] = nextNoteAlphas[i] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                }
                for (i = 0; i < d_absolutePosition; i++){
                        g = calculateGradientWithRespectTo(absolutePosAlphas, i, settings, startIndex, endIndex);
                        m_absolutePos[i] = beta_1 * m_absolutePos[i] + beta_3 * g;
                        v_absolutePos[i] = beta_2 * v_absolutePos[i] + beta_4 * pow(g, 2);
                        m_hat = m_absolutePos[i] / (1 - pow(beta_1, time));
                        v_hat = v_absolutePos[i] / (1 - pow(beta_2, time));
                        absolutePosAlphas[i] = absolutePosAlphas[i] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                }

                //Connecting layer
                for (i = 0; i < d_connectingLayer; i++){
                    for (j = 0; j < d_embedding; j++){
                        g = calculateGradientWithRespectTo(connectingLayerWeights[0][i], j, settings, startIndex, endIndex);
                        m_connectingLayerWeights[0][i][j] = beta_1 * m_velocityEmbedding[i][j] + beta_3 * g;
                        v_connectingLayerWeights[0][i][j] = beta_2 * v_velocityEmbedding[i][j] + beta_4 * pow(g, 2);
                        m_hat = m_connectingLayerWeights[0][i][j] / (1 - pow(beta_1, time));
                        v_hat = v_connectingLayerWeights[0][i][j] / (1 - pow(beta_2, time));
                        connectingLayerWeights[0][i][j] = connectingLayerWeights[i][0][j] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                    }
                }
                for (i = 0; i < d_model; i++){
                    for (j = 0; j < d_connectingLayer; j++){
                        g = calculateGradientWithRespectTo(connectingLayerWeights[1][i], j, settings, startIndex, endIndex);
                        m_connectingLayerWeights[1][i][j] = beta_1 * m_velocityEmbedding[i][j] + beta_3 * g;
                        v_connectingLayerWeights[1][i][j] = beta_2 * v_velocityEmbedding[i][j] + beta_4 * pow(g, 2);
                        m_hat = m_connectingLayerWeights[1][i][j] / (1 - pow(beta_1, time));
                        v_hat = v_connectingLayerWeights[1][i][j] / (1 - pow(beta_2, time));
                        connectingLayerWeights[1][i][j] = connectingLayerWeights[i][0][j] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                    }
                }

                for (i = 0; i < d_connectingLayer; i++){
                        g = calculateGradientWithRespectTo(connectingLayerBiases, i, settings, startIndex, endIndex);
                        m_connectingLayerBiases[i] = beta_1 * m_connectingLayerBiases[i] + beta_3 * g;
                        v_connectingLayerBiases[i] = beta_2 * v_connectingLayerBiases[i] + beta_4 * pow(g, 2);
                        m_hat = m_connectingLayerBiases[i] / (1 - pow(beta_1, time));
                        v_hat = v_connectingLayerBiases[i] / (1 - pow(beta_2, time));
                        connectingLayerBiases[i] = connectingLayerBiases[i] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                }

                //FFN weights and biases
                for (i = 0; i < layers; i++){
                    for (j = 0; j < d_ffn; j++){
                        for (k = 0; k < d_model; k++){
                            g = calculateGradientWithRespectTo(ffnWeights[i][0][j], k, settings, startIndex, endIndex);
                            m_ffnWeights[i][0][j][k] = beta_1 * m_ffnWeights[i][0][j][k] + beta_3 * g;
                            v_ffnWeights[i][0][j][k] = beta_2 * v_ffnWeights[i][0][j][k] + beta_4 * pow(g, 2);
                            m_hat = m_ffnWeights[i][0][j][k] / (1 - pow(beta_1, time));
                            v_hat = v_ffnWeights[i][0][j][k] / (1 - pow(beta_2, time));
                            ffnWeights[i][0][j][k] = ffnWeights[i][0][j][k] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                        }
                    }
                    for (j = 0; j < d_model; j++){
                        for (k = 0; k < d_ffn; k++){
                            g = calculateGradientWithRespectTo(ffnWeights[i][1][j], k, settings, startIndex, endIndex);
                            m_ffnWeights[i][1][j][k] = beta_1 * m_ffnWeights[i][0][j][k] + beta_3 * g;
                            v_ffnWeights[i][1][j][k] = beta_2 * v_ffnWeights[i][0][j][k] + beta_4 * pow(g, 2);
                            m_hat = m_ffnWeights[i][1][j][k] / (1 - pow(beta_1, time));
                            v_hat = v_ffnWeights[i][1][j][k] / (1 - pow(beta_2, time));
                            ffnWeights[i][1][j][k] = ffnWeights[i][1][j][k] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                        }
                    }
                    for (j = 0; j < d_ffn; j++){
                        g = calculateGradientWithRespectTo(ffnBiases[i], j, settings, startIndex, endIndex);
                        m_ffnBiases[i][j] = beta_1 * m_ffnBiases[i][j] + beta_3 * g;
                        v_ffnBiases[i][j] = beta_2 * v_ffnBiases[i][j] + beta_4 * pow(g, 2);
                        m_hat = m_ffnBiases[i][j] / (1 - pow(beta_1, time));
                        v_hat = v_ffnBiases[i][j] / (1 - pow(beta_2, time));
                        ffnBiases[i][j] = ffnBiases[i][j] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                    }
                }

                //Attention matricies
                for (i = 0; i < layers; i++){
                    for (j = 0; j < headsPerLayer; j++){
                        for (k = 0; k < d_attention; k++){
                            for (l = 0; l < d_model; l++){
                                g = calculateGradientWithRespectTo(keyMatricies[i][j][k], l, settings, startIndex, endIndex);
                                m_keyMatricies[i][j][k][l] = beta_1 * m_keyMatricies[i][j][k][l] + beta_3 * g;
                                v_keyMatricies[i][j][k][l] = beta_2 * v_keyMatricies[i][j][k][l] + beta_4 * pow(g, 2);
                                m_hat = 0 / (1 - pow(beta_1, time));
                                v_hat = 0 / (1 - pow(beta_2, time));
                                keyMatricies[i][j][k][l] = keyMatricies[i][j][k][l] - m_hat * (alpha / (sqrt(v_hat) + epsilon));

                                g = calculateGradientWithRespectTo(quarryMatricies[i][j][k], l, settings, startIndex, endIndex);
                                m_quarryMatricies[i][j][k][l] = beta_1 * m_quarryMatricies[i][j][k][l] + beta_3 * g;
                                v_quarryMatricies[i][j][k][l] = beta_2 * v_quarryMatricies[i][j][k][l] + beta_4 * pow(g, 2);
                                m_hat = 0 / (1 - pow(beta_1, time));
                                v_hat = 0 / (1 - pow(beta_2, time));
                                quarryMatricies[i][j][k][l] = quarryMatricies[i][j][k][l] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                                
                                g = calculateGradientWithRespectTo(valueUpMatricies[i][j][k], l, settings, startIndex, endIndex);
                                m_valueUpMatricies[i][j][k][l] = beta_1 * m_valueUpMatricies[i][j][k][l] + beta_3 * g;
                                v_valueUpMatricies[i][j][k][l] = beta_2 * v_valueUpMatricies[i][j][k][l] + beta_4 * pow(g, 2);
                                m_hat = 0 / (1 - pow(beta_1, time));
                                v_hat = 0 / (1 - pow(beta_2, time));
                                valueUpMatricies[i][j][k][l] = valueUpMatricies[i][j][k][l] - m_hat * (alpha / (sqrt(v_hat) + epsilon));

                                g = calculateGradientWithRespectTo(valueDownMatricies[i][j][k], l, settings, startIndex, endIndex);
                                m_valueDownMatricies[i][j][k][l] = beta_1 * m_valueDownMatricies[i][j][k][l] + beta_3 * g;
                                v_valueDownMatricies[i][j][k][l] = beta_2 * v_valueDownMatricies[i][j][k][l] + beta_4 * pow(g, 2);
                                m_hat = 0 / (1 - pow(beta_1, time));
                                v_hat = 0 / (1 - pow(beta_2, time));
                                valueDownMatricies[i][j][k][l] = valueDownMatricies[i][j][k][l] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                            }
                        }
                    }
                }

                //Unembedding
                for (i = 0; i < keyRange + velocityRange + 3; i++){
                    for (j = 0; j < d_model; j++){
                        g = calculateGradientWithRespectTo(unembeddingMatrix[i], j, settings, startIndex, endIndex);
                        m_unembeddingMatrix[i][j] = beta_1 * m_unembeddingMatrix[i][j] + beta_3 * g;
                        v_unembeddingMatrix[i][j] = beta_2 * v_unembeddingMatrix[i][j] + beta_4 * pow(g, 2);
                        m_hat = 0 / (1 - pow(beta_1, time));
                        v_hat = 0 / (1 - pow(beta_2, time));
                        unembeddingMatrix[i][j] = unembeddingMatrix[i][j] - m_hat * (alpha / (sqrt(v_hat) + epsilon));
                    }
                }
            }
        }
#pragma region Deallocation
    //Embedding matricies
    for (i = 0; i < keyRange; i++) {
        delete[] m_keyEmbedding[i];
        delete[] v_keyEmbedding[i];
    }
    delete[] m_keyEmbedding;
    delete[] v_keyEmbedding;

    for (i = 0; i < velocityRange; i++) {
        delete[] m_velocityEmbedding[i];
        delete[] v_velocityEmbedding[i];
    }
    delete[] m_velocityEmbedding;
    delete[] v_velocityEmbedding;

    //Embedding alphas
    delete[] m_prevNoteAlpha;
    delete[] v_prevNoteAlpha;
    delete[] m_nextNoteAlpha;
    delete[] v_nextNoteAlpha;
    delete[] m_absolutePos;
    delete[] v_absolutePos;

    //Connecting layer
    for (i = 0; i < d_connectingLayer; i++) {
        delete[] m_connectingLayerWeights[0][i];
        delete[] v_connectingLayerWeights[0][i];
    }
    for (i = 0; i < d_model; i++) {
        delete[] m_connectingLayerWeights[1][i];
        delete[] v_connectingLayerWeights[1][i];
    }
    delete[] m_connectingLayerWeights[0];
    delete[] v_connectingLayerWeights[0];
    delete[] m_connectingLayerWeights[1];
    delete[] v_connectingLayerWeights[1];
    delete[] m_connectingLayerWeights;
    delete[] v_connectingLayerWeights;

    delete[] m_connectingLayerBiases;
    delete[] v_connectingLayerBiases;

    //FFN weights and biases
    for (i = 0; i < layers; i++) {
        for (j = 0; j < d_ffn; j++) {
            delete[] m_ffnWeights[i][0][j];
            delete[] v_ffnWeights[i][0][j];
        }
        for (j = 0; j < d_model; j++) {
            delete[] m_ffnWeights[i][1][j];
            delete[] v_ffnWeights[i][1][j];
        }
        delete[] m_ffnWeights[i][0];
        delete[] v_ffnWeights[i][0];
        delete[] m_ffnWeights[i][1];
        delete[] v_ffnWeights[i][1];
        delete[] m_ffnBiases[i];
        delete[] v_ffnBiases[i];
    }
    delete[] m_ffnWeights;
    delete[] v_ffnWeights;
    delete[] m_ffnBiases;
    delete[] v_ffnBiases;

    // Attention matricies
    for (i = 0; i < layers; i++) {
        for (j = 0; j < headsPerLayer; j++) {
            for (k = 0; k < d_attention; k++) {
                delete[] m_keyMatricies[i][j][k];
                delete[] v_keyMatricies[i][j][k];
                delete[] m_quarryMatricies[i][j][k];
                delete[] v_quarryMatricies[i][j][k];
                delete[] m_valueUpMatricies[i][j][k];
                delete[] v_valueUpMatricies[i][j][k];
                delete[] m_valueDownMatricies[i][j][k];
                delete[] v_valueDownMatricies[i][j][k];
            }
            delete[] m_keyMatricies[i][j];
            delete[] v_keyMatricies[i][j];
            delete[] m_quarryMatricies[i][j];
            delete[] v_quarryMatricies[i][j];
            delete[] m_valueUpMatricies[i][j];
            delete[] v_valueUpMatricies[i][j];
            delete[] m_valueDownMatricies[i][j];
            delete[] v_valueDownMatricies[i][j];
        }
        delete[] m_keyMatricies[i];
        delete[] v_keyMatricies[i];
        delete[] m_quarryMatricies[i];
        delete[] v_quarryMatricies[i];
        delete[] m_valueUpMatricies[i];
        delete[] v_valueUpMatricies[i];
        delete[] m_valueDownMatricies[i];
        delete[] v_valueDownMatricies[i];
    }
    delete[] m_keyMatricies;
    delete[] v_keyMatricies;
    delete[] m_quarryMatricies;
    delete[] v_quarryMatricies;
    delete[] m_valueUpMatricies;
    delete[] v_valueUpMatricies;
    delete[] m_valueDownMatricies;
    delete[] v_valueDownMatricies;

    // Unembedding
    for (i = 0; i < keyRange + velocityRange + 3; i++) {
        delete[] m_unembeddingMatrix[i];
        delete[] v_unembeddingMatrix[i];
    }
    delete[] m_unembeddingMatrix;
    delete[] v_unembeddingMatrix;
#pragma endregion
    }


float NoteTransformer::calculateCost(int** input, float** expectedOutput){
    float cost  = 0;
    int j;
    float** recieved = process(input);
    for (int i = 0; i < contextSize; i++){
        for (j = 0; j < keyRange; j++){
            cost += pow((recieved[i][j] - expectedOutput[i][j]), 2);
        }
        for (j = keyRange; j < velocityRange; j++){
            cost += pow((recieved[i][j] - expectedOutput[i][j]), 2);
        }
        for (j = keyRange + velocityRange; j < keyRange + velocityRange + 3; j++){
            cost += abs((recieved[i][j] - expectedOutput[i][j]));
        }
    }
    for (j = 0; j < contextSize; j++){
        delete[] recieved[j];
    }
    delete[] recieved;
    return cost;
}

float NoteTransformer::calculateGradientWithRespectTo(float* array, int index, TrainingSettings settings, int startIndex, int endIndex){
    float nudge = 0.000001;
    float originalWeight = array[index];
    array[index] = originalWeight + nudge;
    float costHigher = calculateAverageCost(settings.getDataPath(), startIndex, endIndex);
    array[index] = originalWeight + nudge;
    float costLower = calculateAverageCost(settings.getDataPath(), startIndex, endIndex);
    array[index] = originalWeight;
    return (float) (costHigher - costLower) / (float) (2.0f * nudge);
}
float NoteTransformer::calculateAverageCost(string dirPath, int startIndex, int endIndex){
    float sum = 0;
    int n = 0;
    for (int i = startIndex; i <= endIndex; i++){
        sum += calculateCost(FileUtils::readIntMatrixFromFile(dirPath.append("input").append(to_string(i))), 
                FileUtils::readFloatMatrixFromFile(dirPath.append("output").append(to_string(i))));
        n++;
    }
    return sum / n;
}

void NoteTransformer::save(string dirPath){
        int i, j;
        string currentPath = dirPath;
        try{
        std::filesystem::create_directories(dirPath);
        //Embedding matricies
        FileUtils::saveFloatMatrixToFiles(dirPath + "/key_embedding", keyEmbeddingMatrix, keyRange, d_keyEmbedding);
        FileUtils::saveFloatMatrixToFiles(dirPath + "/velocity_embedding", velocityEmbeddingMatrix, velocityRange, d_velocityEmbedding);
        //Embedding alphas
        FileUtils::saveFloatVectorToFiles(dirPath + "/prev_note_alphas", prevNoteAlphas, d_prevNoteEmbedding);
        FileUtils::saveFloatVectorToFiles(dirPath + "/next_note_alphas", nextNoteAlphas, d_nextNoteEmbedding);
        FileUtils::saveFloatVectorToFiles(dirPath + "/abs_pos_alphas", absolutePosAlphas, d_absolutePosition);
        //Connecting layer
        string currentPath = dirPath + "/connecting_layer";
        std::filesystem::create_directory(currentPath);
        FileUtils::saveFloatMatrixToFiles(currentPath + "/connection0", connectingLayerWeights[0], d_connectingLayer, d_embedding);
        FileUtils::saveFloatMatrixToFiles(currentPath + "/connection1", connectingLayerWeights[1], d_model, d_connectingLayer);
        FileUtils::saveFloatVectorToFiles(currentPath + "/biases", connectingLayerBiases, d_connectingLayer);
        //FFN weights and biases
        for (i = 0; i < layers; i++){
            currentPath = dirPath + "/layers/layer" + to_string(i) + "/ffn_weights";
            std::filesystem::create_directories(currentPath);
            FileUtils::saveFloatMatrixToFiles(currentPath + "/connection0", ffnWeights[i][0], d_ffn, d_model);
            FileUtils::saveFloatMatrixToFiles(currentPath + "/connection1", ffnWeights[i][1], d_model, d_ffn);
        }
        FileUtils::saveFloatMatrixToFiles(dirPath + "/ffn_biases", ffnBiases, layers, d_ffn);
        //Attention matricies
        for (i = 0; i < layers; i++){
            for (j = 0; j < headsPerLayer; j++){
                currentPath = dirPath + "/layers/layer" + to_string(i) + "/attention/head" + to_string(j);
                std::filesystem::create_directories(currentPath);
                FileUtils::saveFloatMatrixToFiles(currentPath + "/keyMatrix", keyMatricies[i][j], d_attention, d_model);
                FileUtils::saveFloatMatrixToFiles(currentPath + "/quarryMatrix", quarryMatricies[i][j], d_attention, d_model);
                FileUtils::saveFloatMatrixToFiles(currentPath + "/valueDownMatrix", valueDownMatricies[i][j], d_attention, d_model);
                FileUtils::saveFloatMatrixToFiles(currentPath + "/ValueUpMatrix", valueUpMatricies[i][j], d_attention, d_model);
            }
        }
        //Unembedding
        FileUtils::saveFloatMatrixToFiles(dirPath + "/unembedding", unembeddingMatrix, keyRange + velocityRange + 3, d_model);
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
        variation = sqrt(2.0f / (float)d_embedding);
        std::normal_distribution<float> distribution6(0, variation);
        for (i = 0; i < d_connectingLayer; i++){
            for (j = 0; j < d_embedding; j++){
                connectingLayerWeights[0][i][j] = distribution6(generator);
            }
        }
        variation = sqrt(2.0f / (float)d_connectingLayer);
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
            variation = sqrt(2.0f / (float)d_model);
            std::normal_distribution<float> distribution8(0, variation);
            for (j = 0; j < d_ffn; j++){
                for (k = 0; k < d_model; k++){
                    ffnWeights[i][0][j][k] = distribution8(generator);
                }
            }
            variation = sqrt(2.0f / (float)d_ffn);
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
        variation = sqrt(2.0f / (float)d_model);
        std::normal_distribution<float> distribution10(0, variation);
        for (i = 0; i < layers; i++){
            for (j = 0; j < headsPerLayer; j++){
                for (k = 0; k < d_attention; k++){
                    for (l = 0; l < d_model; l++){
                        keyMatricies[i][j][k][l] = distribution10(generator);
                        quarryMatricies[i][j][k][l] = distribution10(generator);
                        valueUpMatricies[i][j][k][l] = distribution10(generator);
                        valueDownMatricies[i][j][k][l] = distribution10(generator);
                    }
                }
            }
        }

        //Unembedding
        variation = sqrt(6.0f /(float)keyRange + (float)velocityRange + 3.0f);
        std::normal_distribution<float> distribution11(-variation, variation);
        for (i = 0; i < keyRange + velocityRange + 3; i++){
            for (j = 0; j < d_model; j++){
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
                valueDownMatricies[i][j] = FileUtils::readFloatMatrixFromFile(currentPath + "/valueDownMatrix");
                valueUpMatricies[i][j] = FileUtils::readFloatMatrixFromFile(currentPath + "/valueUpMatrix");
            }
        }

        //Unembedding
        unembeddingMatrix = FileUtils::readFloatMatrixFromFile(dirPath + "/unembedding");
    }

int NoteTransformer::getNumberOfParameters(){
    int params = 0;
    //Embedding matricies
    params += keyRange * d_keyEmbedding + velocityRange * d_velocityEmbedding;

    //Embedding alphas
    params += d_prevNoteEmbedding + d_nextNoteEmbedding + d_absolutePosition;

    //Connecting
    params += d_embedding * d_connectingLayer + d_connectingLayer * d_model + d_connectingLayer;

    //FFN
    params += layers * (d_model *d_ffn * 2 + d_ffn);

    //Attention
    params += layers * 4 * d_attention *d_model;

    //Unembedding
    params += d_model * (keyRange + velocityRange + 3);
    return params;
}

NoteTransformer::~NoteTransformer() {
    int i, j, k; 
    delete[] prevNoteAlphas;
    delete[] nextNoteAlphas;
    delete[] absolutePosAlphas;
    delete[] connectingLayerBiases;

    //Embedding matricies
    for (i = 0; i < keyRange; i++) {
        delete[] keyEmbeddingMatrix[i];
    }
    delete[] keyEmbeddingMatrix;

    for (i = 0; i < velocityRange; i++) {
        delete[] velocityEmbeddingMatrix[i];
    }
    delete[] velocityEmbeddingMatrix;

    //Connecting layer
    for (i = 0; i < d_connectingLayer; i++) {
        delete[] connectingLayerWeights[0][i];
    }
    delete[] connectingLayerWeights[0];

    for (i = 0; i < d_model; i++) {
        delete[] connectingLayerWeights[1][i];
    }
    delete[] connectingLayerWeights[1];
    delete[] connectingLayerWeights;

    //FFN weights and biases
    for (i = 0; i < layers; i++) {
        for (k = 0; k < d_ffn; k++) {
            delete[] ffnWeights[i][0][k];
        }
        delete[] ffnWeights[i][0];
        for (k = 0; k < d_model; k++) {
            delete[] ffnWeights[i][1][k];
        }
        delete[] ffnWeights[i][1];
        delete[] ffnBiases[i];
        delete[] ffnWeights[i];
    }
    delete[] ffnWeights;
    delete[] ffnBiases;

    //Attention matricies
    for (i = 0; i < layers; i++) {
        for (j = 0; j < headsPerLayer; j++) {
            for (k = 0; k < d_attention; k++) {
                delete[] quarryMatricies[i][j][k];
                delete[] keyMatricies[i][j][k];
                delete[] valueDownMatricies[i][j][k];
                delete[] valueUpMatricies[i][j][k];
            }

            delete[] quarryMatricies[i][j];
            delete[] keyMatricies[i][j];
            delete[] valueUpMatricies[i][j];
            delete[] valueDownMatricies[i][j];
        }

        delete[] quarryMatricies[i];
        delete[] keyMatricies[i];
        delete[] valueUpMatricies[i];
        delete[] valueDownMatricies[i];
    }
    delete[] quarryMatricies;
    delete[] keyMatricies;
    delete[] valueUpMatricies;
    delete[] valueDownMatricies;

    //Unembedding
    for (i = 0; i < keyRange + velocityRange + 3; i++) {
        delete[] unembeddingMatrix[i];
    }
    delete[] unembeddingMatrix;
}
