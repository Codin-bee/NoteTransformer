#ifndef MLP_H
#define MLP_H

class MLP
{
private:
    int inputLength, outputLength;
    int* hiddenLayersLengths;
    int numHiddenLayers;
public:
    /**MLP 
     * @param input Number of neurons in input layer
     * @param output Number of neurons in output
     * @param hiddens Lengths of hidden layers
     * @param numHiddens Number of hidden layers
     */
    MLP(int input, int output, int hiddens[], int numHiddens)
        : inputLength(input), outputLength(output), numHiddenLayers(numHiddens)

    {
        hiddenLayersLengths = new int[numHiddenLayers];
        for (int i = 0; i < numHiddenLayers; ++i) {
            hiddenLayersLengths[i] = hiddens[i];
        }
    }

    ~MLP() {
        delete[] hiddenLayersLengths;
    }

    void setInputLength(int input) {
        inputLength = input;
    }

    int getInputLength() {
        return inputLength;
    }

    void setOutputLength(int output) {
        outputLength = output;
    }

    int getOutputLength() {
        return outputLength;
    }

};
#endif