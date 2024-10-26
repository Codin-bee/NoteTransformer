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
    
    for (int i = contextSize; i < length; i++) {
        output = VarUtils::floorAndCastToInt(transformer.process(input), contextSize, 5);
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