# HistAware - NLP pipeline

Mining Historical Trajectories of Awareness: A machine learning approach to historicized sentiment mining.

## Project phases:

1. **Text selection**
   1. Naïve text selection using a keyword search and a synonym search that leverages the dutch NLP model by Spacy.
   2. This text selection is then divided by decades, in order to get a more accurate representation of potential concept drift of the words in the articles. In this step, the articles are also divided into paragraph of maximum 510 words, as the next step does not allow more than that.
   3. Using the dataset filtered by the naïve serach, for each decade we carry out non-naïve text selection that uses a similarity ranking between selected sentences and the remaining corpus.
2. **Sentiment analysis**
   1. Using the filtered dataset from point **1** and divided by decades, we now test sentiment analysis algorithms:
      1. First, we use out-of-the-box cased transformers models pre-trained on Dutch (Bertje, multilingual, XXX).
      2. Then, we select 10/20% of the dataset from point _1.3_ and, after classifying it manually, we turn to fine-tuning the model.

## Project toolset

TODO

## Project documentation

TODO
