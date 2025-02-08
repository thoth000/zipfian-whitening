cd models
gunzip GoogleNews-vectors-negative300.bin.gz
wget https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip
rm crawl-300d-2M.vec.zip
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
unzip crawl-300d-2M-subword.zip
rm crawl-300d-2M-subword.zip
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz
gunzip cc.ja.300.vec.gz

cd ..
python src/convert_model_to_torch.py