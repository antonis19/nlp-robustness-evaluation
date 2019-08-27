# Robustness Evaluation for NLP

### Download Data
Run `sh data/download_data.sh` to download the data needed for the experiments. 
This will download the preprocessed IMDB dataset, the word embeddings for the neural network and for the adversarial attack, and dictionaries of cached nearest neighbors.

### Download StanfordCoreNLP
Run `sh download_stanfordcorenlp.sh` to download the Standford CoreNLP parser.

### Train LSTM model
Run `python train_lstm_model.py` to train an LSTM model for sentiment classification on the IMDB dataset.


### Perform verification using DeepGo
Run `python verification.py` to perform robustness analysis using [DeepGo](https://arxiv.org/abs/1805.02242) on the LSTM. 

### Obtain explanations for sentiment analysis.
Run the code in `sbe_examples.ipynb` to generate explanations for sentiment analysis, using the adaptation of [Spectrum-Based Explanations](https://arxiv.org/abs/1908.02374v1) to text classification. 

### Generate adversarial examples
Run the code in `attack.ipynb` to generate adversarial examples for sentiment analysis.
To visualize the generated adversarial examples run the code in `visualize_attack.ipynb`. 
Some examples of adversarial examples are already shown in `visualize_attack.ipynb`.

### Fix the classification
Run the code in `fix_classification.ipynb` to generate suggestions that change the classification
of a text to the correct class. To visualize the suggestions run the code in `visualize_fixing.ipynb`.

