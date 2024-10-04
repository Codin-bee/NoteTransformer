# Note Transformer
.....some text soon™.....
Modified transformer architecture, implemented from scratch, modified and train to be able to generate simple piano music.

## Coventions

### Naming
* Dimensions: d_ _index name_ (e.g. d_model)
* Constants: uppercase with underscores for spaces (e.g. MY_CONSTANT)
* Normal variables: camel case (e.g. myVariable)
* Filesystem names: snake case (e.g. my_directory)
* MLP/FFN: multi-layer perceptrons and feed forward networks are refferrd to as FFN (uppercase)
* Greek letters: english notation (e.g. alpha, beta)
* Hyperparameters: always the one letter notation instead the full name(e.g. m, v)
* Indexes, superscripts: var_ _index_ (e.g. beta_1, m_hat)
* Attention matricies: _ussage_ Matricies (e.g. keyMatricies, valueDownMatricies)
### Indexing
* Normal matricies: matrix[collum][row]
* Attention matricies: keyMatrix[layer][head][collum][row]
