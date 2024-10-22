#include <NoteTransformer.h>
#ifndef CALCULATOR_H
#define CALCULATOR_H
class Calculator{
public:
    /// @brief Calculates how many trainable parameters would Note Transformer contain in case it had given size and dimensions
    /// @param parameters The sizes and dimensions of the Note Transformer
    /// @return Number of trainable parameters in the Note Transformer
    static int numberOfParameters(ntParams parameters);

    /// @brief Estimates how much time will it take to train given NoteTransformer
    /// @param parameters Sizes and dimensions of given Note Transformer
    /// @param iterrations Number of epochs(iterrations) during the training process
    /// @param examples Number of examples the model is trained on (batchs size * number of batches)
    /// @param passTime Average time it takes to pass one example trought the NoteTransformer in ms
    /// @return The estimated time for the training in ms (note: this does not consider the operations done for calculating the weight
    ///         changes, because there is no good way to estimate it and it is usually really small fraction of training time)
    static double trainingTime(ntParams parameters, int iterrations, int examples, double passTime);
};
#endif