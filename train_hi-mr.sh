#!/usr/bin/env bash

#$ -N pgn-himr
#$ -wd /export/b08/nbafna1/projects/pgns-for-lrmt/
#$ -m e
#$ -t 1-3
#$ -j y -o qsub_logs/pgnhimr_$TASK_ID.out

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=20G,mem_free=30G,gpu=1,hostname=b1[123456789]|c0*|c1[123456789]

# Submit to GPU queue
#$ -q g.q

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
source /home/gqin2/scripts/acquire-gpu 1

conda activate pgnenv
cd /export/b08/nbafna1/projects/pgns-for-lrmt/


epochs_all=(40 30 20)
epochs=${epochs_all[$SGE_TASK_ID-1]}
batch_size=12
max_lines_all=(15000 30000 60000)
max_lines=${max_lines_all[$SGE_TASK_ID-1]}

EXP_ID="pgn"
MODEL_NAME="$EXP_ID-hi~mr-epochs~$epochs-max_lines~$max_lines"
TOKENIZER_NAME="$EXP_ID-hi~mr-max_lines~$max_lines"

MODEL_OUTPUT_DIR="models/$MODEL_NAME"
TOKENIZER_INPATH="tokenizers/$TOKENIZER_NAME"
LOG_DIR="logs/$MODEL_NAME"
mkdir -p "tokenizers/"
mkdir -p $MODEL_OUTPUT_DIR
mkdir -p $LOG_DIR


python pgn_scratch.py \
--DATADIR_L1 /export/b08/nbafna1/data/cvit-pib-v1.3/hi-mr/splits/hi/ \
--DATADIR_L2 /export/b08/nbafna1/data/cvit-pib-v1.3/hi-mr/splits/mr/ \
--TOKENIZER_INPATH $TOKENIZER_INPATH \
--OUTPUT_DIR $MODEL_OUTPUT_DIR --LOG_DIR $LOG_DIR --epochs $epochs --batch_size $batch_size \
--max_lines $max_lines


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