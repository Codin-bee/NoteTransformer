#include <NoteTransformer.h>
#ifndef CALCULATOR_H
#define CALCULATOR_H
class Calculator{
public:
    int numberOfParameters(ntParams parameters);
    double trainingTime(ntParams parameters, int iterrations, int examples, double passTime);
};
#endif