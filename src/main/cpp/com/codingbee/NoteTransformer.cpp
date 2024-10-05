#include "NoteTransformer.h"
#include "FileUtils.h"
#include <random>

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
        for (int epoch = 0; epoch < settings.getEpochs(); epoch++){
            for(int batchNo = 0; batchNo < FileUtils::getNumberOfFilesInDir(settings.getDataPath()) / settings.getBatchSize(); batchNo++){
                int startIndex = batchNo * settings.getBatchSize();
                int endIndex = startIndex + settings.getBatchSize();
                //~for every batch looping every parameter

                //Embedding matricies
                for (i = 0; i < keyRange; i++){
                    for (j = 0; j < d_keyEmbedding; j++){
                        keyEmbeddingMatrix[i][j] = 0;
                    }
                }
                for (i = 0; i < velocityRange; i++){
                    for (j = 0; j < d_velocityEmbedding; j++){
                        velocityEmbeddingMatrix[i][j] = 0;
                    }
                }

                //Embedding aplhas
                for (i = 0; i < d_prevNoteEmbedding; i++){
                        prevNoteAlphas[i] = 0;
                }
                for (i = 0; i < d_nextNoteEmbedding; i++){
                        nextNoteAlphas[i] = 0;
                }
                for (i = 0; i < d_absolutePosition; i++){
                        absolutePosAlphas[i] = 0;
                }

                //Connecting layer
                for (i = 0; i < d_connectingLayer; i++){
                    for (j = 0; j < d_embedding; j++){
                       connectingLayerWeights[0][i][j] = 0;
                    }
                }
                for (i = 0; i < d_model; i++){
                    for (j = 0; j < d_connectingLayer; j++){
                        connectingLayerWeights[1][i][j] = 0;
                    }
                }

                for (i = 0; i < d_connectingLayer; i++){
                    connectingLayerBiases[i] = 0;
                }

                //FFN weights and biases
                for (i = 0; i < layers; i++){
                    for (j = 0; j < d_ffn; j++){
                        for (k = 0; k < d_model; k++){
                            ffnWeights[i][0][j][k] = 0;
                        }
                    }
                    for (j = 0; j < d_model; j++){
                        for (k = 0; k < d_ffn; k++){
                            ffnWeights[i][1][j][k] = 0;
                        }
                    }
                    for (j = 0; j < d_ffn; j++){
                        ffnBiases[i][j] = 0;
                    }
                }

                //Attention matricies
                for (i = 0; i < layers; i++){
                    for (j = 0; j < headsPerLayer; j++){
                        for (k = 0; k < d_attention; k++){
                            for (l = 0; l < d_model; l++){
                                keyMatricies[i][j][k][l] = 0;
                                quarryMatricies[i][j][k][l] = 0;
                                valueUpMatricies[i][j][k][l] = 0;
                                valueDownMatricies[i][j][k][l] = 0;
                            }
                        }
                    }
                }

                //Unembedding
                for (i = 0; i < keyRange + velocityRange + 3; i++){
                    for (j = 0; j < d_prevNoteEmbedding; j++){
                       unembeddingMatrix[i][j] = 0;
                    }
                }
            }
        }
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
        string currentPath;

        //Embedding matricies
        FileUtils::saveFloatMatrixToFiles(dirPath + "/key_embedding", keyEmbeddingMatrix, keyRange, d_keyEmbedding);
        FileUtils::saveFloatMatrixToFiles(dirPath + "/velocity_embedding", velocityEmbeddingMatrix, velocityRange, d_velocityEmbedding);

        //Embedding alphas
        FileUtils::saveFloatVectorToFiles(dirPath + "/prev_note_alphas", prevNoteAlphas, d_prevNoteEmbedding);
        FileUtils::saveFloatVectorToFiles(dirPath + "/next_note_alphas", nextNoteAlphas, d_nextNoteEmbedding);
        FileUtils::saveFloatVectorToFiles(dirPath + "/abs_pos_alphas", absolutePosAlphas, d_absolutePosition);

        //Connecting layer
        currentPath = dirPath + "/connecting_layer";
        FileUtils::saveFloatMatrixToFiles(currentPath + "/connection0", connectingLayerWeights[0], d_connectingLayer, d_embedding);
        FileUtils::saveFloatMatrixToFiles(currentPath + "/connection1", connectingLayerWeights[1], d_embedding, d_connectingLayer);
        FileUtils::saveFloatVectorToFiles(currentPath + "/biases", connectingLayerBiases, d_connectingLayer);

        //FFN weights and biases
        for (i = 0; i < layers; i++){
            currentPath = dirPath + "/layers/layer" + to_string(i) + "/ffn_weights";
            FileUtils::saveFloatMatrixToFiles(currentPath + "/connection0", ffnWeights[i][0], d_ffn, d_model);
            FileUtils::saveFloatMatrixToFiles(currentPath + "/connection1", ffnWeights[i][1], d_model, d_ffn);
        }
        FileUtils::saveFloatMatrixToFiles(dirPath + "/ffn_biases", ffnBiases, layers, d_attention);

        //Attention matricies
        for (i = 0; i < layers; i++){
            for (j = 0; j < headsPerLayer; j++){
                currentPath = dirPath + "/layers/layer" + to_string(i) + "/attention/head" + to_string(j);
                FileUtils::saveFloatMatrixToFiles(currentPath + "keyMatrix", keyMatricies[i][j], d_attention, d_model);
                FileUtils::saveFloatMatrixToFiles(currentPath + "quarryMatrix", quarryMatricies[i][j], d_attention, d_model);
                FileUtils::saveFloatMatrixToFiles(currentPath + "valueDownMatrix", valueDownMatricies[i][j], d_attention, d_model);
                FileUtils::saveFloatMatrixToFiles(currentPath + "ValueUpMatrix", valueUpMatricies[i][j], d_attention, d_model);
            }
        }

        //Unembedding
        FileUtils::saveFloatMatrixToFiles(dirPath + "/unembedding", unembeddingMatrix, keyRange + velocityRange + 3, d_model);
    }

void NoteTransformer::randomInit(){
        allocateModelMemory();
        int i, j, k, l;

        std::normal_distribution<float>;// distribution(0, 1);
        std::default_random_engine generator;
        float variation;

        //Embeding matricies
        variation = sqrt(6 / (keyRange * d_keyEmbedding));
        std::normal_distribution<float> distribution(-variation, variation);
        for (i = 0; i < keyRange; i++){
            for (j = 0; j < d_keyEmbedding; j++){
                keyEmbeddingMatrix[i][j] = distribution(generator);
            }
        }
        variation = sqrt(6 / (velocityRange * d_velocityEmbedding));
        std::normal_distribution<float> distribution(-variation, variation);
        for (i = 0; i < velocityRange; i++){
            for (j = 0; j < d_velocityEmbedding; j++){
                velocityEmbeddingMatrix[i][j] = distribution(generator);
            }
        }

        //Embedding aplhas
        variation = sqrt(d_nextNoteEmbedding);
        std::normal_distribution<float> distribution(-variation, variation);
        for (i = 0; i < d_prevNoteEmbedding; i++){
            prevNoteAlphas[i] = distribution(generator);
        }
        variation = sqrt(d_prevNoteEmbedding);
        std::normal_distribution<float> distribution(-variation, variation);
        for (i = 0; i < d_nextNoteEmbedding; i++){
            nextNoteAlphas[i] = distribution(generator);
        }
        variation = sqrt(d_absolutePosition);
        std::normal_distribution<float> distribution(-variation, variation);
        for (i = 0; i < d_absolutePosition; i++){
            absolutePosAlphas[i] = distribution(generator);
        }

        //Connecting layer
        variation = sqrt(2 / d_embedding);
        std::normal_distribution<float> distribution(0, variation);
        for (i = 0; i < d_connectingLayer; i++){
            for (j = 0; j < d_embedding; j++){
                connectingLayerWeights[0][i][j] = distribution(generator);
            }
        }
        variation = sqrt(2 / d_connectingLayer);
        std::normal_distribution<float> distribution(0, variation);
        for (i = 0; i < d_model; i++){
            for (j = 0; j < d_connectingLayer; j++){
                connectingLayerWeights[1][i][j] = distribution(generator);
            }
        }

        for (i = 0; i < d_connectingLayer; i++){
            connectingLayerBiases[i] = 0;
        }

        //FFN weights and biases
        for (i = 0; i < layers; i++){
            variation = sqrt(2 / d_model);
            std::normal_distribution<float> distribution(0, variation);
            for (j = 0; j < d_ffn; j++){
                for (k = 0; k < d_model; k++){
                    ffnWeights[i][0][j][k] = distribution(generator);
                }
            }
            variation = sqrt(2 / d_ffn);
            std::normal_distribution<float> distribution(0, variation);
            for (j = 0; j < d_model; j++){
                for (k = 0; k < d_ffn; k++){
                    ffnWeights[i][1][j][k] = distribution(generator);
                }
            }
            for (j = 0; j < d_ffn; j++){
                ffnBiases[i][j] = 0;
            }
        }

        //Attention matricies
        variation = sqrt(2 / d_model);
        std::normal_distribution<float> distribution(0, variation);
        for (i = 0; i < layers; i++){
            for (j = 0; j < headsPerLayer; j++){
                for (k = 0; k < d_attention; k++){
                    for (l = 0; l < d_model; l++){
                        keyMatricies[i][j][k][l] = distribution(generator);
                        quarryMatricies[i][j][k][l] = distribution(generator);
                        valueUpMatricies[i][j][k][l] = distribution(generator);
                        valueDownMatricies[i][j][k][l] = distribution(generator);
                    }
                }
            }
        }

        //Unembedding
        variation = sqrt(6 / keyRange + velocityRange + 3);
        std::normal_distribution<float> distribution(-variation, variation);
        for (i = 0; i < keyRange + velocityRange + 3; i++){
            for (j = 0; j < d_prevNoteEmbedding; j++){
                unembeddingMatrix[i][j] = distribution(generator);
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
                keyMatricies[i][j] = FileUtils::readFloatMatrixFromFile(currentPath + "keyMatrix");
                quarryMatricies[i][j] = FileUtils::readFloatMatrixFromFile(currentPath + "quarryMatrix");
                valueDownMatricies[i][j] = FileUtils::readFloatMatrixFromFile(currentPath + "valueDownMatrix");
                valueUpMatricies[i][j] = FileUtils::readFloatMatrixFromFile(currentPath + "valueUpMatrix");
            }
        }

        //Unembedding
        unembeddingMatrix = FileUtils::readFloatMatrixFromFile(dirPath + "/unembedding");
    }


NoteTransformer::~NoteTransformer(){
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
                delete[] ffnBiases[i];

                for (k = 0; k < d_ffn; k++){
                    delete[] ffnWeights[i][0][k];
                }

                for (k = 0; k < d_model; k++){
                    delete[] ffnWeights[i][1][k];
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