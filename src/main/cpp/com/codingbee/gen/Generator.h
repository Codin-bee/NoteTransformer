#ifndef GENERATOR_H
#define GENERATOR_H
#include <NoteTransformer.h>


class Generator{
    public:
     static int** newTrack(NoteTransformer& transformer, int** initialSequence, int length, bool mask, int velocityFadeLength);

    private: 
     static int** convertOutputToNotes(float** output, int contextSize);
};

#endif