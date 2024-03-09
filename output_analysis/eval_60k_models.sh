#!/usr/bin/env bash

#$ -N pgneval
#$ -wd /export/b08/nbafna1/projects/pgns-for-lrmt/
#$ -m e
#$ -t 1
#$ -j y -o output_analysis/qsub_logs/pgneval_$TASK_ID.out

# Fill out RAM/memory (same thing) request,
# the number of GPUs you want,
# and the hostnames of the machines for special GPU models.
#$ -l ram_free=10G,mem_free=10G,gpu=1,hostname=!c08*&!c07*&!c03*&!c04*&!c25*&c*

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

batch_size=32

EXP_ID="pgn_eval_60k_run_$SGE_TASK_ID"

PROJ_DIR="/export/b08/nbafna1/projects/pgns-for-lrmt/"

MODEL_NAMES=("pgn-es~ca-wm-epochs~20-max_lines~60000-vocab_size~16000" \
"pgn-fr~oc-wm-epochs~20-max_lines~60000-vocab_size~16000" \
"pgn-hi~bh-nllb-epochs~40-max_lines~60000-vocab_size~16000" \
"pgn-hi~mr-epochs~20-max_lines~60000" \
"vanilla-es~ca-wm-epochs~20-max_lines~60000-vocab_size~16000" \
"vanilla-fr~oc-wm-epochs~20-max_lines~60000-vocab_size~16000" \
"vanilla-hi~bh-nllb-epochs~40-max_lines~60000-vocab_size~16000" \
"vanilla-hi~mr-epochs~20-max_lines~60000")

TOKENIZER_NAMES=("pgn-es~ca-wm-max_lines~60000-vocab_size~16000" \
"pgn-fr~oc-wm-max_lines~60000-vocab_size~16000" \
"pgn-hi~bh-nllb-max_lines~60000-vocab_size~16000" \
"pgn-hi~mr-max_lines~60000" \
"vanilla-es~ca-wm-max_lines~60000-vocab_size~16000" \
"vanilla-fr~oc-wm-max_lines~60000-vocab_size~16000" \
"vanilla-hi~bh-nllb-max_lines~60000-vocab_size~16000" \
"vanilla-hi~mr-max_lines~60000")

MODEL_NAME=${MODEL_NAMES[$SGE_TASK_ID-1]}
TOKENIZER_NAME=${TOKENIZER_NAMES[$SGE_TASK_ID-1]}

MODEL_OUTPUT_DIR="$PROJ_DIR/models/$MODEL_NAME"
TOKENIZER_INPATH="$PROJ_DIR/tokenizers/$TOKENIZER_NAME"
OUTPUT_DIR="$PROJ_DIR/output_analysis/output_translations/$MODEL_NAME"
mkdir -p $OUTPUT_DIR

DATAFILE_L1S=("/export/b08/nbafna1/data/wikimatrix/es-ca/splits/es/test" \
"/export/b08/nbafna1/data/wikimatrix/fr-oc/splits/fr/test" \
"/export/b08/nbafna1/data/nllb/hin-bho/splits/hin/test" \
"/export/b08/nbafna1/data/cvit-pib-v1.3/hi-mr/splits/hi/test" \
"/export/b08/nbafna1/data/wikimatrix/es-ca/splits/es/test" \
"/export/b08/nbafna1/data/wikimatrix/fr-oc/splits/fr/test" \
"/export/b08/nbafna1/data/nllb/hin-bho/splits/hin/test" \
"/export/b08/nbafna1/data/cvit-pib-v1.3/hi-mr/splits/hi/test")

DATAFILE_L2S=("/export/b08/nbafna1/data/wikimatrix/es-ca/splits/ca/test" \
"/export/b08/nbafna1/data/wikimatrix/fr-oc/splits/oc/test" \
"/export/b08/nbafna1/data/nllb/hin-bho/splits/bho/test" \
"/export/b08/nbafna1/data/cvit-pib-v1.3/hi-mr/splits/mr/test" \
"/export/b08/nbafna1/data/wikimatrix/es-ca/splits/ca/test" \
"/export/b08/nbafna1/data/wikimatrix/fr-oc/splits/oc/test" \
"/export/b08/nbafna1/data/nllb/hin-bho/splits/bho/test" \
"/export/b08/nbafna1/data/cvit-pib-v1.3/hi-mr/splits/mr/test")

DATAFILE_L1=${DATAFILE_L1S[$SGE_TASK_ID-1]}
DATAFILE_L2=${DATAFILE_L2S[$SGE_TASK_ID-1]}

python $PROJ_DIR/evaluation.py \
--DATAFILE_L1 $DATAFILE_L1 \
--DATAFILE_L2 $DATAFILE_L2 \
--MODEL_INPATH $MODEL_OUTPUT_DIR \
--TOKENIZER_INPATH $TOKENIZER_INPATH \
--output_dir $OUTPUT_DIR \
--EXP_ID $EXP_ID \
--save_results




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
