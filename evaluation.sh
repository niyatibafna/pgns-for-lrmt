#!/usr/bin/env bash

#$ -N basic
#$ -wd /export/b08/nbafna1/projects/mt_hw_skeleton/
#$ -m e
#$ -t 1
#$ -j y -o qsub_logs/expname_$TASK_ID.out

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=20G,mem_free=30G,gpu=1,hostname=b1[123456789]|c0*|c1[123456789]

# Submit to GPU queue
#$ -q g.q

# Assign a free-GPU to your program (make sure -n matches the requested number of GPUs above)
source /home/gqin2/scripts/acquire-gpu 1
conda activate pgenv
cd /export/b08/nbafna1/projects/mt_hf_skeleton/

EXP_ID="basic"
epochs=10
max_lines=15000
MODEL_NAME="$EXP_ID~l1-l2-epochs~$epochs-max_lines~$max_lines"
TOKENIZER_NAME="$EXP_ID~l1-l2~max_lines-$max_lines"

MODEL_OUTPUT_DIR="models/$MODEL_NAME"
TOKENIZER_INPATH="tokenizers/$TOKENIZER_NAME"

python evaluation.py \
--DATAFILE_L1 /export/b08/nbafna1/projects/pointer-networks-for-same-family-nmt/data/europarl.es-ca.es/test \
--DATAFILE_L2 /export/b08/nbafna1/projects/pointer-networks-for-same-family-nmt/data/europarl.es-ca.ca/test \
--MODEL_INPATH $MODEL_OUTPUT_DIR \
--TOKENIZER_INPATH $TOKENIZER_INPATH \
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
