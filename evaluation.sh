#!/usr/bin/env bash

#$ -N pgneval
#$ -wd /export/b08/nbafna1/projects/pgns-for-lrmt/
#$ -m e
#$ -t 1-3
#$ -j y -o qsub_logs/pgneval_$TASK_ID.out

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=10G,mem_free=10G,gpu=1,hostname=b1[123456789]|c0*|c1[123456789]

# Submit to GPU queue
#$ -q g.q

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
source /home/gqin2/scripts/acquire-gpu 1
conda activate pgnenv
cd /export/b08/nbafna1/projects/pgns-for-lrmt/


epochs_all=(40 30 20)
epochs=${epochs_all[$SGE_TASK_ID-1]}
batch_size=32
max_lines_all=(15000 30000 60000)
max_lines=${max_lines_all[$SGE_TASK_ID-1]}

EXP_ID="pgn"
# MODEL_NAME="$EXP_ID-es~ca-epochs~$epochs-max_lines~$max_lines"
MODEL_NAME="pgn-es~ca-epochs~$epochs-max_lines~$max_lines/"
TOKENIZER_NAME="$EXP_ID-es~ca-max_lines~$max_lines"

MODEL_OUTPUT_DIR="models/$MODEL_NAME"
TOKENIZER_INPATH="tokenizers/$TOKENIZER_NAME"
OUTPUT_DIR="output_translations/$MODEL_NAME"
mkdir -p $OUTPUT_DIR

python evaluation.py \
--DATAFILE_L1 /export/b08/nbafna1/data/europarl.es-ca/splits/es/test \
--DATAFILE_L2 /export/b08/nbafna1/data/europarl.es-ca/splits/ca/test \
--MODEL_INPATH $MODEL_OUTPUT_DIR \
--TOKENIZER_INPATH $TOKENIZER_INPATH \
--output_dir $OUTPUT_DIR \
--EXP_ID $EXP_ID 




# parser.add_argument("--DATAFILE_L1", type=str, required=True, help="Path to the eval data directory")
#     parser.add_argument("--DATAFILE_L2", type=str, required=True, help="Path to the eval data directory")
#     parser.add_argument("--TOKENIZER_INPATH", type=str, default=None, help="Path to the tokenizer")
#     parser.add_argument("--MODEL_INPATH", type=str, default=None, help="Path to the model")
#     parser.add_argument("--EXP_ID", type=str, required=True, help="Experiment ID")
#     parser.add_argument("--save_results", action="store_true", help="Save results to file")
#     parser.add_argument("--task", type=str, default="translation", help="Task to evaluate")
#     # Accept transformed_datapath, from_file,
#     parser.add_argument("--output_dir", type=str, default=None, help="Path to the transformed data")
#     parser.add_argument("--from_file", action="store_true", help="Evaluate from file")
