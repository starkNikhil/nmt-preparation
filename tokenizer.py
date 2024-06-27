#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Training BPE models for the source and target
# Command: python3 train.py <train_source_file_tok> <train_target_file_tok>

import sys
from tokenizers import Tokenizer, trainers, models, pre_tokenizers, decoders, processors

path = ""  # change the path if needed

train_source_file_tok = path + sys.argv[1]
train_target_file_tok = path + sys.argv[2]

def train_bpe_tokenizer(input_file, model_prefix, vocab_size=50000):
    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Customize the pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Setup the trainer
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=[
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

    # Train the tokenizer
    tokenizer.train(files=[input_file], trainer=trainer)

    # Save the tokenizer
    tokenizer.save(f"{model_prefix}.json")

# Train the BPE tokenizer for the source language
train_bpe_tokenizer(train_source_file_tok, "source")
print("Done, training a BPE tokenizer for the Source finished successfully!")

# Train the BPE tokenizer for the target language
train_bpe_tokenizer(train_target_file_tok, "target")
print("Done, training a BPE tokenizer for the Target finished successfully!")
