This is the code for [Pointer-Generator Networks for Low-Resource Machine Translation: Don't Copy That!](https://arxiv.org/abs/2403.10963), presented at LREC-COLING '24.

Train MT with PGNs like this:
```
python pgn.py \
--DATADIR_L1 /export/b08/nbafna1/data/europarl.es-ca/splits/es/ \
--DATADIR_L2 /export/b08/nbafna1/data/europarl.es-ca/splits/ca/ \
--TOKENIZER_INPATH $TOKENIZER_INPATH \
--OUTPUT_DIR $MODEL_OUTPUT_DIR --LOG_DIR $LOG_DIR --epochs $epochs --batch_size $batch_size \
--max_lines $max_lines
```
`$TOKENIZER_INPATH` is the path to a HuggingFace `AutoTokenizer`, `L1` and `L2` are the source and target languages respectively.
