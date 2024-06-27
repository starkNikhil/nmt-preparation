import sys
import sentencepiece as spm

path = ""  # change the path if needed

train_source_file_tok = path + sys.argv[1]
train_target_file_tok = path + sys.argv[2]

def train_sentencepiece_model(input_file, model_prefix, vocab_size):
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        # Removed user_defined_symbols as these are already defined in control_symbols
        # user_defined_symbols=['<pad>', '<unk>', '<s>', '</s>'],
        hard_vocab_limit=False
    )
    print(f"Done, training SentencePiece BPE model for {model_prefix} finished successfully!")

# Train the source subword model
train_sentencepiece_model(train_source_file_tok, "source", 50000)

# Train the target subword model
train_sentencepiece_model(train_target_file_tok, "target", 50000)
