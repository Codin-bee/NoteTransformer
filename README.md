# Note Transformer
A custom Transformer architecture designed to generate simple piano music, built from scratch to reduce the number of parameters while maintaining high performance. It features custom embeddings, a specialized layer for handling timing, and a custom unembedding process. Future updates will incorporate multiple datasets to improve generalization and music quality.

## Coventions

### Naming
* Dimensions: d_ _index name_ (e.g. d_model)
* Constants: uppercase with underscores for spaces (e.g. MY_CONSTANT)
* Normal variables: camel case (e.g. myVariable)
* Filesystem names: snake case (e.g. my_directory)
* MLP/FFN: multi-layer perceptrons and feed forward networks are refferrd to as ffn (lowercase)
* Greek letters: english notation (e.g. alpha, beta)
* Hyperparameters: always the one letter notation instead the full name(e.g. m, v)
* Indexes, superscripts: var_ _index_ (e.g. beta_1, m_hat)
* Attention matricies: _ussage_ Matricies (e.g. keyMatricies, valueDownMatricies)
### Indexing
* Normal matricies: matrix[collum][row]
* Attention matricies: keyMatrix[layer][head][collum][row]
* Weights: weights[layer][connection][neuronTo][neuronFrom]
### File System
All parameters are stored inside .txt files given by the 
* Embedding vectors and matricies: ~/ _part_ _embedding (e.g. key_embedding, velocity_embedding, prev_note_alphas)
* Connecting layer: ~/connecting_layer/ _part_ (e.g. ~/connecting_layer/connection0, ~/connecting_layer/biases)
* FFN weights: ~/layers/layerN/ffn_weights/connectionN (e.g. ~/layers/layer1/ffn_weights/connection0)
* FFN biase: ~/ffn_biases
* Attention: ~/layers/layerN/attention/headN/ _use_ matrix (e.g. layers/layer1/attention/head1/keyMatrix)
* Unembedding: ~/unembedding
