from datasets import load_dataset
import sys

sys.path.insert(1, DIR_HELPER)

import ai_helper as ai_h
import constants_dataset as c_d

model_checkpoint = "microsoft/swin-tiny-patch4-window7-224" # pre-trained model from which to fine-tune
#model_checkpoint = "microsoft/swin-base-patch4-window7-224-in22k" # pre-trained model from which to fine-tune
batch_size = 32 # batch size for training and evaluation


# load a custom dataset from local/remote files or folders using the ImageFolder feature

# option 1: local/remote files (supporting the following formats: tar, gzip, zip, xz, rar, zstd)
dataset = load_dataset(c_d.DIR_DATASET_EUROSAT, data_files="https://madm.dfki.de/files/sentinel/EuroSAT.zip")