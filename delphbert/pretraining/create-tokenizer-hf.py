from glob import glob

from tokenizers import BertWordPieceTokenizer

NUM_THREADS = 8
VOCABSIZE = 32_000
NUM_SENTS = 100_000_000

SOURCE_PATH = '/home/leonardovida/data/volume_1/data-histaware/merged_articles/1970s/merged_articles/test/*.txt'

input_paths = list(glob(SOURCE_PATH))
input_path = ','.join(input_paths)

print('Total number of files: {}'.format(len(input_paths)))

# Cased model

tokenizer = BertWordPieceTokenizer(
    clean_text=True, 
    handle_chinese_chars=False,
    strip_accents=False,
    lowercase=False, 
)

trainer = tokenizer.train( 
    files=input_paths,
    vocab_size=32000,
    min_frequency=2,
    show_progress=True,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
    limit_alphabet=1000,
    wordpieces_prefix="##"
)

tokenizer.save("/home/leonardovida/data/volume_1/data-histaware/tokenizer/dutch.bert.cased.128.json")

# Uncased model

# tokenizer = BertWordPieceTokenizer(
#     clean_text=True,
#     handle_chinese_chars=False,
#     strip_accents=False,  # We need to investigate that further (stripping helps?)
#     lowercase=True,
# )

# trainer = tokenizer.train(
#     files=input_paths,
#     vocab_size=32000,
#     min_frequency=2,
#     show_progress=True,
#     special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'],
#     limit_alphabet=1000,
#     wordpieces_prefix="##"
# )

# tokenizer.save("/home/leonardovida/data/volume_1/data-histaware/tokenizer/uncased")