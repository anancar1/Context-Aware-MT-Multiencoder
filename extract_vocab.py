import tensorflow as tf
import tensorflow_datasets as tfds
import os
import numpy as np
import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create SubwordTextEncoder for languages')
    parser.add_argument('-p', '--path', help='Path where traing sentences are  (src,tgt,context)')
    parser.add_argument('-s', '--src', help='Source language code')
    parser.add_argument('-t', '--tgt', help='Target language code')
    args = parser.parse_args()
    vars = vars(args)
    path = vars['path']
    src = vars['src']
    tgt =vars['tgt']

    train_src = []
    with open(f'{path}/train.{src}') as f:
        lines = f.read().split("\n")[:-1]
    for line in lines: 
        train_src.append(line)

    train_tgt = []
    with open(f'{path}/train.{tgt}') as f:
        lines = f.read().split("\n")[:-1]
    for line in lines: 
        train_tgt.append(line)

    train_pairs = []
    for i in range(len(train_src)):
        train_pairs.append((train_src[i],train_tgt[i]))

    tokenizer_src = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (s for s, t in train_pairs), target_vocab_size=2**13)

    tokenizer_tgt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        (t for s, t in train_pairs), target_vocab_size=2**13)

    tokenizer_src.save_to_file(f"{path}/vocab_encoder_{src}")
    tokenizer_tgt.save_to_file(f"{path}/vocab_encoder_{tgt}")