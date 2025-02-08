import gensim
import os
from sentence_transformers.models import WordEmbeddings

"""
This scripts converts word2vec/ fasttext models to torch format.
This is needed to integrate the word2vec model with the SentenceTransformer library.
You need to run only once.
"""
W2V_BIN_PATH = "models/GoogleNews-vectors-negative300.bin"
W2V_TXT_PATH = "models/GoogleNews-vectors-negative300.txt"
W2V_VOCAB_PATH = "models/GoogleNews-vectors-negative300.vocab"
W2V_TORCH_PATH = "models/GoogleNews-vectors-negative300-torch"

FASTTEXT_EN_TXT_PATH = "models/crawl-300d-2M.vec"
FASTTEXT_EN_SUBWORD_TXT_PATH = "models/crawl-300d-2M-subword.vec"
FASTTEXT_JA_TXT_PATH = "models/cc.ja.300.vec"

if __name__ == "__main__":
    print("Loading word2vec Model...")
    model = gensim.models.KeyedVectors.load_word2vec_format(W2V_BIN_PATH, binary=True)
    print("Saving word2vec Model...")
    model.save_word2vec_format(W2V_TXT_PATH, fvocab=W2V_VOCAB_PATH, binary=False)
    print("Converting model to pytorch format...")
    word_embedding = WordEmbeddings.from_text_file(W2V_TXT_PATH)
    os.makedirs(W2V_TORCH_PATH, exist_ok=True)
    word_embedding.save(W2V_TORCH_PATH)

    # fasttext as torch
    print("Loading fasttext EN Model...")
    word_embedding = WordEmbeddings.from_text_file(FASTTEXT_EN_TXT_PATH)
    os.makedirs("models/fasttext-en-torch", exist_ok=True)
    word_embedding.save("models/fasttext-en-torch")

    # fasttext as torch
    print("Loading fasttext EN Model...")
    word_embedding = WordEmbeddings.from_text_file(FASTTEXT_EN_SUBWORD_TXT_PATH)
    os.makedirs("models/fasttext-en-subword-torch", exist_ok=True)
    word_embedding.save("models/fasttext-en-subword-torch")

    print("Loading fasttext JA Model...")
    word_embedding = WordEmbeddings.from_text_file(FASTTEXT_JA_TXT_PATH)
    os.makedirs("models/fasttext-ja-torch", exist_ok=True)
    word_embedding.save("models/fasttext-ja-torch")
