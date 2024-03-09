#!/usr/bin/env bash

#$ -N eval-esca
#$ -wd /export/b08/nbafna1/projects/pgns-for-lrmt/
#$ -m e
#$ -t 4
#$ -j y -o output_analysis/qsub_logs/evalff_$TASK_ID.out

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=10G,mem_free=10G

source ~/.bashrc
conda deactivate
conda activate pgnenv

cd /export/b08/nbafna1/projects/pgns-for-lrmt/

set -x # print out every command that's run with a +

epochs_all=(50 50 30 20)
epochs=${epochs_all[$SGE_TASK_ID-1]}
batch_size=12
max_lines_all=(5000 15000 30000 60000)
max_lines=${max_lines_all[$SGE_TASK_ID-1]}
vocab_size=16000

PROJ_DIR="/export/b08/nbafna1/projects/pgns-for-lrmt/"

EXP_ID="pgn"
# EXP_ID="vanilla"
MODEL_NAME="$EXP_ID-es~ca-wm-epochs~$epochs-max_lines~$max_lines-vocab_size~$vocab_size"
FILES_DIR="$PROJ_DIR/output_analysis/logs/$MODEL_NAME"
inputs_file="$FILES_DIR/inputs.txt"
references_file="$FILES_DIR/references.txt"
predictions_file="$FILES_DIR/predictions.txt"

python $PROJ_DIR/evaluation.py \
--DATAFILE_L1 $inputs_file \
--DATAFILE_L2 $references_file \
--predictions_file $predictions_file \
--from_file \
--EXP_ID $EXP_ID \
--save_results