#include "Generator.h"
#include "Exception.h"
#include <MemoryUtils.h>
#include <VarUtils.h>

using namespace std;

int** Generator::newTrack(NoteTransformer& transformer, int** initialSequence, int length, bool mask, int velocityFadeLength) {
    int contextSize = transformer.getContextSize();
    int** track = new int*[length];
    
    for (int i = 0; i < length; i++) {
        track[i] = new int[5];
        if (i < contextSize) {
            for (int j = 0; j < 5; j++) {
                track[i][j] = initialSequence[i][j];
            }
        }
    }

    int** input = VarUtils::floorAndCastToInt(transformer.process(initialSequence), contextSize, 5);
    if (!input) {
        throw Exception("The function call returned null pointer", ExceptionType::INNER_ERROR);
    }

    int** output = nullptr;
    
    for (int i = contextSize - 1; i < length; i++) {
        output = convertOutputToNotes(transformer.process(input), contextSize);
        if (!output) {
            throw Exception("The function call returned null pointer", ExceptionType::INNER_ERROR);
        }
        for (int j = 0; j < 5; j++) {
            track[i][j] = output[contextSize - 1][j];
        }
        if (mask) {
            output[contextSize - 1][3] = 0;
        }
        MemoryUtils::deallocateMatrix(input, contextSize);
        input = output;

        if (i >= length - velocityFadeLength) {
            input[contextSize - 1][2] = input[contextSize - 1][2] * 0.8;
        }
    }

    MemoryUtils::deallocateMatrix(input, contextSize);
    return track;
}

int **Generator::convertOutputToNotes(float **output, int contextSize){
    int** notes = new int*[contextSize];
    for (int i = 0; i < contextSize; i++){
        notes[i] = new int[5];
        notes[i][0] = VarUtils::getHighestIndexInSubVector(output[i], 0, 127);
        notes[i][1] = VarUtils::getHighestIndexInSubVector(output[i], 128, 255) - 128;
        notes[i][2] = output[i][256];
        notes[i][3] = output[i][257];
        notes[i][4] = output[i][258];
    }
    return notes;
}
