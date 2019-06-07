#!/usr/bin/env python
import os
import subprocess
import numpy as np
from tqdm.auto import tqdm


n_sample_sets = 100
num_neurons = 12
small_sample_size = 2**20
sample_size = small_sample_size*n_sample_sets

out_dir = os.path.join('samples fc',
                       'dataset_maxent_num_samples_{}_num_neurons_{}_num_bins_1_critic_iters_5_lambda_10.0_num_layers_2_num_units_512_iteration_0'.format(small_sample_size, num_neurons))

sample = np.zeros((sample_size, num_neurons), dtype=np.int)


FNULL = open(os.devnull, 'w')

for s in tqdm(range(n_sample_sets)):
    subprocess.run("python main_conv.py --architecture='fc' --dataset='maxent' --num_bins=1 --num_neurons={}  --num_samples={}".format(num_neurons, small_sample_size), stdout=FNULL, stderr=FNULL, shell=True, check=True)
    with np.load(os.path.join(out_dir, 'samples_fake.npz')) as data:
        binarized = (data['samples'] > np.random.random(data['samples'].shape)).astype(np.int)
    sample[s*small_sample_size:(s+1)*small_sample_size,:] = binarized.T
np.savetxt(os.path.join(out_dir, 'sample_{}.txt'.format(sample_size)), sample, fmt='%d')
