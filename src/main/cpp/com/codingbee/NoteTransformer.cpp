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
                this->coonectLayer(embeddedMatrix[i], processedMatrix[i], i);
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
        return finalOutput;
}