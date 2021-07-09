### To finish
* Use trained tokenizer to create pretrain data and not out-of-the box one

### Requirements
* Python 3.7.10
* tensorflow-1.15--GPU

### Notes
* Interesting guide: https://github.com/stefan-it/turkish-bert/blob/master/CHEATSHEET.md
  * Minimum: 60 GB RAM 


## Guide

### Prepare data

**Run with Python 3.8.5**

* Put the `merged_files` into one folder, let's call it `raw_files`
* Run `KB-prepare-newspaper.py` giving the `raw_files` as input and a destination folder, let's call it `text_files`, as output
* After running you should have many text files each of 500k (or less) lines

### Create Tokenizer

**Run with Python 3.8.5**

* Run `create_vocab.py` on the entire `text_files`
* Once it's saved you should have 2 files:
  * `xxx.model`
  * `xxx.vocab`
* Run `vocab.py` with `xxx.vocab` as input and `xxx.vocab.mod` as output
* Now put the three files (or just `xxx.model` and `xxx.vocab.mod`) in a separate folder called `tokenizer`

### Preprocessing

**Switch to Python 3.8.5**

* Clone: git clone https://github.com/google-research/bert.git

* In `create_pretrain_data.py` put 
  * for VOCAB_PATH the `xxx.vocab.mod` file path
  * for SOURCE_PATH all the txt files (check if you don't have to split them before)
  * for DEST_PATHS the folder of destination of the processed data

Run

* You might want to shuffle the data after creating it (use the `shuffle-pretraining-data.py`)

**Important**
Set the specs of your model NOW.
Both here and in `config.py`

### Training

**Switch to Python 3.7.10 with Tensorflow 1.15-gpu**

* Modify the KB-run-pretraining-google accordingly and run

**Remember to match the specs of the bert model that you set beforehand!**