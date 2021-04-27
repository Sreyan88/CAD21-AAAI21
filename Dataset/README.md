# Dataset

The folder *Original_Dataset* contains the dataset provided by the organiser. Each training instance is a complete slide with all the tokens present in the slide. 

Statistics of the Original Dataset is as follows:

|      |Total Slides | Total Sentences | Total Tokens |
|  ---  | ----------- | ---------------- | ------------- |
|Train | 1241        |    8849         |    96934     |
|Dev   | 180         |    1175         |    12822     |
|Test  | 355         |    2569         |    28108     |


The original dataset that was provided by the organisers do not contain the POS tags, to use POS tags as a feature for the model POS tags were added to the original datasets. The folder *Dataset_with_POS* contains the dataset with POS tags as one feature.

A feature which depicts if a token is scientific keyword or not which was used as an additional feature in our paper. The folder *Dataset_with_POS,Keyphrase* contains the datasets with POS and Keyphrase features.

The folder *Dataset_POS,PUNCT* contains the dataset where POS tags and whether a token is punctuation are used as feaures.
