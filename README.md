# HistAware - NLP pipeline

Mining Historical Trajectories of Awareness: A machine learning approach to historicized sentiment mining.

## Project phases

The pipleine

### Pipeline description

1. **Text selection**:
   1. Delpher texts are parsed
   2. Naïve text selection using a keyword search and a synonym search that leverages the dutch NLP model by Spacy.
      1. **TODO** _1 + 2_: Are they scalable to 000s of millions of texts? Or does the entire pipeline have to work on decades for the next selection step?
      2. The articles are also divided into paragraphs, what is the maximum? It needs to be of 510 words, as the next step does not allow more than that.
   3. Subdivision:
      1. **TODO**: If not already done, text selection is then divided by decades, in order to get a more accurate representation of potential concept drift of the words in the articles.
   4. Using the dataset filtered by the naïve serach, for each decade we carry out non-naïve text selection that uses a similarity ranking between selected sentences and the remaining corpus.
      1. Clean the text from stop words + **names** + **numbers**
      2. Clean the text from words that were wrongly recognized:
         1. Identify them with [MASK] and then use BertJe to fill them the best as possible [https://medium.com/states-title/using-nlp-bert-to-improve-ocr-accuracy-385c98ae174c]
      3. Compare sentence-transformer &/or naive bayes & tfidf & bm25 to select similar texts.
         1. **TODO**: up until which cut-off point? Is it better to have many false postives or false negatives but higher accuracy?
         2. What about using topic modeling?
2. **Sentiment analysis**
   1. Using the filtered dataset from point **1** and divided by decades, we now test sentiment analysis algorithms:
      1. First, we use out-of-the-box cased transformers models pre-trained on Dutch (Bertje, multilingual, XXX).
      2. Then, we select 10/20% of the dataset from point _1.3_ and, after classifying it manually, we turn to fine-tuning the model.

## Usage

WIP
