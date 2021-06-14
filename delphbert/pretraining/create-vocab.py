from glob import glob

from sentencepiece import SentencePieceTrainer


NUM_THREADS = 8
VOCABSIZE = 32_000
NUM_SENTS = 100_000_000


SOURCE_PATH = '/home/leonardovida/data/volume_1/delphbert-results/1-raw_files/1960/*.txt'

input_paths = list(glob(SOURCE_PATH))
input_path = ','.join(input_paths)

print('Total number of files: {}'.format(len(input_paths)))

cmd = f'--input_sentence_size=10000000 --input={input_path} --vocab_size={VOCABSIZE} --num_threads={NUM_THREADS} --shuffle_input_sentence=true --model_type=unigram --split_by_number=false --split_by_unicode_script=false --model_prefix=dutch --bos_piece=[CLS] --eos_piece=[SEP] --unk_piece=[UNK] --control_symbols=[PAD],[MASK]'
trainer = SentencePieceTrainer.Train(cmd)
