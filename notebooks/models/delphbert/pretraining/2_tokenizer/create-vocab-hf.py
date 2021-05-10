from glob import glob

from sentencepiece import SentencePieceTrainer
from tokenizers import BertWordPieceTokenizer

NUM_THREADS = 8
VOCABSIZE = 33_000
NUM_SENTS = 100_000_000


#SOURCE_PATH = '/home/s2971992/Bertje/clean-data-v2/*/*.txt'
SOURCE_PATH = '/home/leonardovida/data/volume_1/data-histaware/merged_articles/1970s/merged_articles/test/*.txt'

input_paths = list(glob(SOURCE_PATH))
input_path = ','.join(input_paths)

print('Total number of files: {}'.format(len(input_paths)))

tokenizer = BertWordPieceTokenizer(lowercase=False, strip_accents=False, clean_text=True)
tokenizer.train(
    files=input_paths,
    vocab_size=VOCABSIZE,
    min_frequency=2,
    show_progress=True,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],limit_alphabet=1000,
    limit_alphabet=1000,
    wordpieces_prefix="##"
)

tokenizer.save_model('/home/leonardovida/data/volume_1/data-histaware/tokenizer')
#cmd = f'--input={input_path} --vocab_size={VOCABSIZE} --num_threads={NUM_THREADS} --input_sentence_size={NUM_SENTS} --shuffle_input_sentence=true --model_type=unigram --split_by_number=false --split_by_unicode_script=false --model_prefix=dutch --bos_piece=[CLS] --eos_piece=[SEP] --unk_piece=[UNK] --control_symbols=[PAD],[MASK]'
#trainer = SentencePieceTrainer.Train(cmd)
