# preparation: calculate the norm of zipf/unif whitened vectors
python src/calculate_uniform_norm.py --model_name models/GoogleNews-vectors-negative300-torch
python src/calculate_uniform_norm.py --model_name sentence-transformers/average_word_embeddings_glove.840B.300d
python src/calculate_uniform_norm.py --model_name models/fasttext-en-torch
python src/calculate_uniform_norm.py --model_naame models/fasttext-en-subword-torch

python src/calculate_zipfian_norm.py --model_name models/GoogleNews-vectors-negative300-torch
python src/calculate_zipfian_norm.py --model_name sentence-transformers/average_word_embeddings_glove.840B.300d
python src/calculate_zipfian_norm.py --model_name models/fasttext-en-torch
python src/calculate_zipfian_norm.py --model_name models/fasttext-en-subword-torch

# run sts evaluation via MTEB
## w/ wiki freq
python src/run_eval.py --model_name models/GoogleNews-vectors-negative300-torch
python src/run_eval.py --model_name sentence-transformers/average_word_embeddings_glove.840B.300d
python src/run_eval.py --model_name models/fasttext-en-torch
python src/run_eval.py --model_name models/fasttext-en-subword-torch
python src/run_eval.py --model_name models/fasttext-ja-torch --task_names "[JSTS]"
## w/ test set freq
python src/run_eval.py --model_name models/GoogleNews-vectors-negative300-torch --in_batch True
python src/run_eval.py --model_name sentence-transformers/average_word_embeddings_glove.840B.300d --in_batch True
python src/run_eval.py --model_name models/fasttext-en-torch --in_batch True
python src/run_eval.py --model_name models/fasttext-en-subword-torch --in_batch True
python src/run_eval.py --model_name models/fasttext-ja-torch --task_names "[JSTS]" --in_batch True

# calculate isotropy scores
python src/calc_isotropy_score.py --model_name models/GoogleNews-vectors-negative300-torch
python src/calc_isotropy_score.py --model_name sentence-transformers/average_word_embeddings_glove.840B.300d 
python src/calc_isotropy_score.py --model_name models/fasttext-en-torch
python src/calc_isotropy_score.py --model_name models/fasttext-en-subword-torch

# run norm experiments
python src/run_norm_experiments.py --model_name sentence-transformers/average_word_embeddings_glove.840B.300d --topk 500
python src/run_norm_experiments.py --model_name models/GoogleNews-vectors-negative300 --topk 500