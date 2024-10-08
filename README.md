# Note Transformer
A custom-built and modified Transformer architecture, implemented from scratch with the goal of reducing the number of parameters while maintaining performance. Custom embeddings were developed, and a specialized layer was added to enhance the model's ability to generate simple piano music. Certain components of the architecture were simplified to further decrease parameter count, optimizing it for more efficient training. While initially trained on a single dataset, future iterations will explore the use of multiple datasets to improve generalization and music quality.

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
* Normal matricies: matrix[row][column]
* Attention matricies: keyMatrix[layer][head][row][column]
