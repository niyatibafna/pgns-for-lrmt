#!/usr/bin/env bash

#$ -N pgn-esen
#$ -wd /export/b08/nbafna1/projects/pgns-for-lrmt/
#$ -m e
#$ -t 1-4
#$ -j y -o qsub_logs/pgnesen_$TASK_ID.out

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=14G,mem_free=14G,gpu=1,hostname=!c08*&!c07*&!c04*&!c25*&c*

# Submit to GPU queue
#$ -q g.q

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
source ~/.bashrc
conda deactivate
conda activate pgnenv

source /home/gqin2/scripts/acquire-gpu 1

cd /export/b08/nbafna1/projects/pgns-for-lrmt/

which python

echo "HOSTNAME: $(hostname)"
echo
echo CUDA in ENV:
env | grep CUDA
echo
echo SGE in ENV:
env | grep SGE

set -x # print out every command that's run with a +
nvidia-smi

epochs_all=(50 50 30 20)
epochs=${epochs_all[$SGE_TASK_ID-1]}
batch_size=12
max_lines_all=(5000 15000 30000 60000)
max_lines=${max_lines_all[$SGE_TASK_ID-1]}
vocab_size=16000

EXP_ID="pgn"
MODEL_NAME="$EXP_ID-es~en-wm-epochs~$epochs-max_lines~$max_lines-vocab_size~$vocab_size"
TOKENIZER_NAME="$EXP_ID-es~en-wm-max_lines~$max_lines-vocab_size~$vocab_size"

MODEL_OUTPUT_DIR="models/$MODEL_NAME"
TOKENIZER_INPATH="tokenizers/$TOKENIZER_NAME"
LOG_DIR="logs/$MODEL_NAME"
mkdir -p "tokenizers/"
mkdir -p $MODEL_OUTPUT_DIR
mkdir -p $LOG_DIR


python pgn_scratch.py \
--DATADIR_L1 /export/b08/nbafna1/data/wikimatrix/es-en/splits/es/ \
--DATADIR_L2 /export/b08/nbafna1/data/wikimatrix/es-en/splits/en/ \
--TOKENIZER_INPATH $TOKENIZER_INPATH \
--OUTPUT_DIR $MODEL_OUTPUT_DIR --LOG_DIR $LOG_DIR --epochs $epochs --batch_size $batch_size \
--max_lines $max_lines --vocab_size $vocab_size
