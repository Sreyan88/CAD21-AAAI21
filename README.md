# Cisco at AAAI-CAD21 shared task: Predicting Emphasis in Presentation Slidesusing Contextualized Embeddings

This repository contains code for our paper titled : [Cisco at AAAI-CAD21 shared task: Predicting Emphasis in Presentation Slidesusing Contextualized Embeddings](https://arxiv.org/pdf/2101.11422.pdf) 

This paper was accepted at 35th AAAI Conference on Artificial Intelligence, and is based on the shared task: Predicting Emphasis in Presentation Slides.

To run the BiLSTM-ELMo and XLNet models, update the files path in *config.py* files and run in cli using
```
python3 run.py
```
Roberta models can be run using provided *iPython Notebooks*

This paper also discusses the effect of POS while predicting emphasis  in presentation slides. POS tags are not part of original datasets. In order to add POS tags to the datasets please use *add_pos_tags.ipynb*. Another extra feature which depicts if a token is a punctutaion or not can be added to the original dataset by running *adding_punctuation.ipynb*.

