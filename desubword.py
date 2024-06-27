#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Decoding the translation prediction using Subword-NMT
# Command: python3 desubword.py <bpe_codes_file> <target_pred_file>

import sys
import codecs
from subword_nmt.apply_bpe import BPE

bpe_codes_file = sys.argv[1]
target_pred = sys.argv[2]
target_decodeded = target_pred + ".desubword"

# Load BPE model
with codecs.open(bpe_codes_file, encoding='utf-8') as codes:
    bpe = BPE(codes)

# Note: Subword-NMT typically uses '@@ ' to indicate subword tokens in the text
def decode_line(line):
    return line.replace('@@ ', '')

with open(target_pred, encoding='utf-8') as pred, open(target_decodeded, "w+", encoding='utf-8') as pred_decoded:
    for line in pred:
        line = line.strip()
        line = decode_line(line)
        pred_decoded.write(line + "\n")
        
print("Done desubwording! Output:", target_decodeded)
