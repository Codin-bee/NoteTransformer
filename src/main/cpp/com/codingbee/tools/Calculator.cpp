#include "Calculator.h"
int Calculator::numberOfParameters(ntParams transformerParameters){
    int paramCount = 0;
    //Embedding matricies
    paramCount += 128 * (transformerParameters.keyDims + transformerParameters.velocityDims);

    //Embedding alphas
    paramCount += transformerParameters.prevDims + transformerParameters.nextDims + transformerParameters.absolutePosDims;

    //Connecting
    paramCount += (transformerParameters.keyDims + transformerParameters.velocityDims + transformerParameters.prevDims + 
     transformerParameters.nextDims + transformerParameters.absolutePosDims)* transformerParameters.connectDims +
     transformerParameters.connectDims * transformerParameters.modelDims + transformerParameters.connectDims;

    //FFN
    paramCount += transformerParameters.layerCount * (transformerParameters.modelDims * transformerParameters.ffnDims *
     2 + transformerParameters.ffnDims);

    //Attention
    paramCount += transformerParameters.layerCount * (2 * (transformerParameters.modelDims / transformerParameters.headsInLayers) *
     transformerParameters.modelDims + transformerParameters.modelDims * transformerParameters.modelDims);

    //Layer normalization
    paramCount += transformerParameters.layerCount * transformerParameters.modelDims * 2;

    //Unembedding
    paramCount += transformerParameters.modelDims * 259;
    return paramCount;
}

double Calculator::trainingTime(ntParams parameters, int iterrations, int examples, double passTime){
    return (double) iterrations * (double) Calculator::numberOfParameters(parameters) * (double) examples * 2 *passTime;
}