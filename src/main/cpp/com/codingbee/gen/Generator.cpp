#include "Generator.h"
#include <VarUtils.h>

int** Generator::newTrack(NoteTransformer& transformer, int** initialSequence, int length, bool mask, bool applyVelocityFade){
    int** track = new int*[length];
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
    for (int i = transformer.getContextSize(); i < length + transformer.getContextSize(); i++){
        output = TypeUtils::floorAndCastToInt(transformer.process(input), transformer.getContextSize(), 5);
        for (int j = 0; j < 5; j++){
            track[i][j] = output[transformer.getContextSize()][j];
        }
        //Mask(opt.)
        if (mask){
            track[i][3] = 0;
        }
        //New input
        
        //Add to sequence
    }

}
