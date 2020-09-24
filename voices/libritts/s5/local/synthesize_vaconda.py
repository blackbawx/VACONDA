""" Synthesis waveform from trained model.

Usage: synthesize_tacotronone.py [options] <checkpoint> <text_list_file> <dst_dir>

options:
    --file-name-suffix=<s>   File name suffix [default: ].
    --max-decoder-steps=<N>  Max decoder steps [default: 500].
    -h, --help               Show help message.

"""
from docopt import docopt

# Use text & audio modules from existing Tacotron implementation.
import sys
import os
from os.path import dirname, join

### This is not supposed to be hardcoded #####
FALCON_DIR = os.environ.get('FALCONDIR')
sys.path.append(FALCON_DIR)

from utils import audio
from utils.plot import plot_alignment

import torch
from torch.autograd import Variable
import numpy as np
import nltk

#from util import *
from model_vqvae import VACONDA

from hyperparameters import hyperparameters

from tqdm import tqdm

import json


use_cuda = torch.cuda.is_available()
hparams = hyperparameters()
vox_dir = 'vox'

def synthesize(model, mspec, spk):
    """Convert text to speech waveform given a Tacotron model.
    """
    if use_cuda:
        model = model.cuda()

    model.eval()

    sequence = np.array(mspec)
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    spk = np.array(spk)
    spk = Variable(torch.from_numpy(spk)).unsqueeze(0)

    if use_cuda:
        sequence = sequence.cuda() 
        spk = spk.cuda()

    with torch.no_grad():
       model.forward_getlatents(sequence)
       mel_outputs, linear_outputs,  = model.forward_eval(sequence, spk)

    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio.denormalize(linear_output)
    waveform = audio.inv_spectrogram(linear_output.T)

    return waveform


if __name__ == "__main__":

    args = docopt(__doc__)
    print("Command line args:\n", args)

    # Override hyper parameters
    #if conf is not None:
    #    with open(conf) as f:
    #        hparams.update_params(f)


    checkpoint_path = args["<checkpoint>"]
    text_list_file_path = args["<text_list_file>"]
    dst_dir = args["<dst_dir>"]
    max_decoder_steps = int(args["--max-decoder-steps"])
    file_name_suffix = args["--file-name-suffix"]

    checkpoint = torch.load(checkpoint_path)
    checkpoints_dir = os.path.dirname(checkpoint_path)
    with open(checkpoints_dir + '/spk_ids') as  f:
       spk_ids = json.load(f)

    model = VACONDA(embedding_dim=256,
                     input_dim=hparams.num_mels,
                     mel_dim = hparams.num_mels,
                     assistant = None,
                     r = hparams.outputs_per_step,
                     use_arff = 0,
                     num_spk = len(spk_ids)
                     )

    checkpoint = torch.load(checkpoint_path)
    checkpoints_dir = os.path.dirname(checkpoint_path)

    model.load_state_dict(checkpoint["state_dict"])

    os.makedirs(dst_dir, exist_ok=True)

    with open(text_list_file_path, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):

            fname = line.decode("utf-8").split()[0]
            cmd = 'cp vox/wav/' + fname + '.wav ' + dst_dir + '/' + fname + '_original.wav'
            print(cmd)
            os.system(cmd)

            mspec_fname = vox_dir + '/festival/falcon_mspec/' + fname + '.feats.npy'
            mspec = np.load(mspec_fname) 
 
            speaker = fname.split('_')[0]
            spk = spk_ids[speaker]

            waveform = synthesize(model, mspec, spk)
            dst_wav_path = join(dst_dir, "{}{}.wav".format(fname + '_generated', file_name_suffix))
            audio.save_wav(waveform, dst_wav_path)

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)

