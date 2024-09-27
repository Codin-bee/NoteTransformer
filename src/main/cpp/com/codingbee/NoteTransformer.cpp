#include "NoteTransformer.h"

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
                this->connectLayer(embeddedMatrix[i], processedMatrix[i], i);
            }));

        }
       for (auto& t : threads){
            if (t.joinable()) {
                t.join();
            }
        }

        
        //Layers
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
            MathUtils::applySoftmax(dotProducts[i], contextSize, softmaxTemperature);
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

void NoteTransformer::ffn(float* vector, int layer, int vectorNo){

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

void NoteTransformer::connectLayer(float* originalVector, float* downscaledVector, int vectorNo){
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

void NoteTransformer::allocateModelMemory(){
        /*TODO: implement memory allocation for all network vectos, matricies and tensors*/
    }

void NoteTransformer::train(TrainingSettings settings){

        for (int i = 0; i < settings.getEpochs(); i++){
            /*TODO: implement training algorithm*/
        }
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


void NoteTransformer::save(string dirPath){
        /*TODO: implement saving matrix into given directory*/
    }

void NoteTransformer::randomInit(){
        allocateModelMemory();
         /*TODO: Implement initialization using random generated values*/
    }

void NoteTransformer::init(string dirPath){
        allocateModelMemory();
        /*TODO: implement initialization from directory*/
    }

