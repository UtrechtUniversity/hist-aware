import os
from glob import glob

from tqdm import tqdm
import tensorflow as tf


FLAGS = {
    'input_file': '/home/leonardovida/data/volume_1/data-histaware/pretraining-data-cased/tf_pretrain_*.tfrecord',
    'output_dir': '/home/leonardovida/data/volume_1/data-histaware/pretraining-data-cased-shuffled',
    'num_examples': 1_000_000, # 10
    'num_shards': 5, # 50
    'seed': 12345,
}

input_files = list(glob(FLAGS['input_file']))

print('Loading, interleaving and shuffling data')
d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
d = d.shuffle(buffer_size=len(input_files), seed=FLAGS['seed'])
d = d.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, sloppy=False, cycle_length=24))
d = d.shuffle(buffer_size=FLAGS['num_examples'], seed=FLAGS['seed'])
print(d)


print('Saving shards to {}'.format(FLAGS['output_dir']))
os.makedirs(FLAGS['output_dir'], exist_ok=True)

num_shards = FLAGS['num_shards']
shard_size = FLAGS['num_examples'] // num_shards


def get_writer(i):
    output_file = os.path.join(FLAGS['output_dir'], 'tf_examples_{}.tfrecord'.format(str(i).zfill(3)))
    return tf.io.TFRecordWriter(output_file)


writer = None
for i, example in tqdm(enumerate(d), total=FLAGS['num_examples']):
    if i % shard_size == 0:
        if writer is not None:
            writer.close()
        writer = get_writer(i // shard_size)
        tqdm.write(' > Start writing shard #{}'.format(i // shard_size))

    writer.write(example.numpy())

writer.close()

print('Done!')
