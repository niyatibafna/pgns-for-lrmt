#!/usr/bin/env bash

#$ -N eval-esen
#$ -wd /export/b08/nbafna1/projects/pgns-for-lrmt/
#$ -m e
#$ -t 4
#$ -j y -o output_analysis/qsub_logs/evaltrainer_$TASK_ID.out

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=10G,mem_free=10G,gpu=1,hostname=!c08*&!c07*&!c04*&!c25*&c*

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

PROJ_DIR="/export/b08/nbafna1/projects/pgns-for-lrmt/"

# EXP_ID="pgn"
EXP_ID="vanilla"

MODEL_NAME="$EXP_ID-es~en-wm-epochs~$epochs-max_lines~$max_lines-vocab_size~$vocab_size"
TOKENIZER_NAME="$EXP_ID-es~en-wm-max_lines~$max_lines-vocab_size~$vocab_size"

MODEL_OUTPUT_DIR="$PROJ_DIR/output_analysis/models/$MODEL_NAME"
PT_CKPT="$PROJ_DIR/models/$MODEL_NAME/"
TOKENIZER_INPATH="$PROJ_DIR/tokenizers/$TOKENIZER_NAME"
LOG_DIR="$PROJ_DIR/output_analysis/logs/$MODEL_NAME"
mkdir -p $MODEL_OUTPUT_DIR
mkdir -p $LOG_DIR


python $PROJ_DIR/pgn_onlyeval.py \
--DATADIR_L1 /export/b08/nbafna1/data/wikimatrix/es-en/splits/es/ \
--DATADIR_L2 /export/b08/nbafna1/data/wikimatrix/es-en/splits/en/ \
--PT_CKPT $PT_CKPT \
--TOKENIZER_INPATH $TOKENIZER_INPATH \
--OUTPUT_DIR $MODEL_OUTPUT_DIR --LOG_DIR $LOG_DIR --epochs $epochs --batch_size $batch_size \
--max_lines 5000 --vocab_size $vocab_size --only_eval


# parser.add_argument("--TRAIN_FILE_L1", type=str, default=None)
# parser.add_argument("--TRAIN_FILE_L2", type=str, default=None)
# parser.add_argument("--ENC_DEC_MODELPATH", type=str, default="bert-base-multilingual-cased", help="Path to encoder model to initalize encoder/decoder (separately)")
# parser.add_argument("--TOKENIZER_INPATH", type=str, default=None, help="Path to tokenizer - if self-trained, put path. If None, \
#                     the tokenizer from the encoder model will be used")
# parser.add_argument("--PT_CKPT", type=str, default=None, help="Path to PGN checkpoint")
# parser.add_argument("--max_length", type=int, default = 512)
# parser.add_argument("--OUTPUT_DIR", type=str, default="output_dir", help="Path to save model")
# parser.add_argument("--LOG_DIR", type=str, default="logs", help="Path to save tensorboard logs")
# parser.add_argument("--epochs", type=int, default = 20)
# parser.add_argument("--batch_size", type=int, default = 16)
# parser.add_argument("--max_lines", type=int, default = INF)
# parser.add_argument("--resume_from_checkpoint", action="store_true", default=False, help="Resume training from args.OUTPUT_DIR")
# parser.add_argument("--force_p_gen", type=float, default = None)