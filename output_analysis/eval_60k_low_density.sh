#!/usr/bin/env bash

#$ -N eval_bottom
#$ -wd /export/b08/nbafna1/projects/pgns-for-lrmt/
#$ -m e
#$ -t 1-12
#$ -j y -o output_analysis/qsub_logs/eval_bottom_$TASK_ID.out

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=10G,mem_free=10G

source ~/.bashrc
conda deactivate
conda activate pgnenv

cd /export/b08/nbafna1/projects/pgns-for-lrmt/

which python

PROJ_DIR="/export/b08/nbafna1/projects/pgns-for-lrmt/"

lang_pairs=("es-ca" "fr-oc" "hi-bh" "hi-mr" "fr-de" "es-en" "es-ca" "fr-oc" "hi-bh" "hi-mr" "fr-de" "es-en")
lang_pair=${lang_pairs[$SGE_TASK_ID-1]}

inputs_file="$PROJ_DIR/output_analysis/test_set_splits_files/$lang_pair/source_bottom_500_sents.txt"
references_file="$PROJ_DIR/output_analysis/test_set_splits_files/$lang_pair/target_bottom_500_sents.txt"

# Now we find the model predictions
MODEL_NAMES=("pgn-es~ca-wm-epochs~20-max_lines~60000-vocab_size~16000" \
"pgn-fr~oc-wm-epochs~20-max_lines~60000-vocab_size~16000" \
"pgn-hi~bh-nllb-epochs~40-max_lines~60000-vocab_size~16000" \
"pgn-hi~mr-epochs~20-max_lines~60000" \
"pgn-fr~de-wm-epochs~20-max_lines~60000-vocab_size~16000" \
"pgn-es~en-wm-epochs~20-max_lines~60000-vocab_size~16000" \
"vanilla-es~ca-wm-epochs~20-max_lines~60000-vocab_size~16000" \
"vanilla-fr~oc-wm-epochs~20-max_lines~60000-vocab_size~16000" \
"vanilla-hi~bh-nllb-epochs~40-max_lines~60000-vocab_size~16000" \
"vanilla-hi~mr-epochs~20-max_lines~60000" \
"vanilla-fr~de-wm-epochs~20-max_lines~60000-vocab_size~16000" \
"vanilla-es~en-wm-epochs~20-max_lines~60000-vocab_size~16000" \
)

MODEL_NAME=${MODEL_NAMES[$SGE_TASK_ID-1]}

predictions_file="$PROJ_DIR/output_analysis/model_output_files/$lang_pair/$MODEL_NAME/bottom_500_sents.txt"

record_file="$PROJ_DIR/output_analysis/results_60k_split_test_sets_low_density.json"

python $PROJ_DIR/evaluation.py \
--DATAFILE_L1 $inputs_file \
--DATAFILE_L2 $references_file \
--predictions_file $predictions_file \
--from_file \
--EXP_ID "low_density_$MODEL_NAME" \
--save_results \
--record_file $record_file

