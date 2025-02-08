MODEL_NAMES="bert-base-uncased FacebookAI/roberta-base microsoft/deberta-base"
POOLING_NAMES="avg_first_last avg_first_last_uniform_centering avg_first_last_uniform_whitening avg_first_last_pseudo-uniform_centering avg_first_last_pseudo-uniform_whitening"
for MODEL_NAME in $MODEL_NAMES; do
    for POOLING_NAME in $POOLING_NAMES; do
        echo "MODEL_NAME: $MODEL_NAME"
        echo "POOLING_NAME: $POOLING_NAME"
        python evaluation.py --model_name_or_path "$MODEL_NAME" --pooler "$POOLING_NAME" --task_set sts --mode test
    done
done