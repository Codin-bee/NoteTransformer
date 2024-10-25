#include "Generator.h"
#include <VarUtils.h>
#include <MemoryUtils.h>

int** Generator::newTrack(NoteTransformer& transformer, int** initialSequence, int length, bool mask, int velocityFadeLength){
    int** track = new int*[length];
    //Initial sequence is added to the track
    for (int i = 0; i < length; i++){
        track[i] = new int[5];
        if (i < transformer.getContextSize()){  
            for (int j = 0; j < 5; j++){
                track[i][j] = initialSequence[i][j];
            }
        }
    }
    int** input = TypeUtils::floorAndCastToInt(transformer.process(initialSequence), transformer.getContextSize(), 5);
    int** output;
    for (int i = transformer.getContextSize(); i < length; i++){
        output = TypeUtils::floorAndCastToInt(transformer.process(input), transformer.getContextSize(), 5);
        for (int j = 0; j < 5; j++){
            //Add note to track
            track[i][j] = output[transformer.getContextSize()][j];
        }
        //Mask(opt.)
        if (mask){
            output[transformer.getContextSize()][3] = 0;
        }
        //New input
        MemoryUtils::deallocateMatrix(input, transformer.getContextSize());
        input = output;
        //Velocity fading for last n notes
        if (i >= length-velocityFadeLength){
            input[transformer.getContextSize()-1][2] *= 0.8;
        }
    }
    return track;
}
