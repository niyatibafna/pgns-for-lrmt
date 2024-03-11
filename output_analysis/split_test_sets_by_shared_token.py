import os
import random

MODEL_NAMES=["-es~ca-wm-epochs~20-max_lines~60000-vocab_size~16000", \
    "-fr~oc-wm-epochs~20-max_lines~60000-vocab_size~16000", \
    "-hi~bh-nllb-epochs~40-max_lines~60000-vocab_size~16000", \
    "-hi~mr-epochs~20-max_lines~60000",\
    "-fr~de-wm-epochs~20-max_lines~60000-vocab_size~16000", \
    "-es~en-wm-epochs~20-max_lines~60000-vocab_size~16000"
    ]
PGN_MODEL_NAMES=["pgn"+model_name for model_name in MODEL_NAMES]
VANILLA_MODEL_NAMES=["vanilla"+model_name for model_name in MODEL_NAMES]

# DATAFILE_L1S=["/export/b08/nbafna1/data/wikimatrix/es-ca/splits/es/test", \
# "/export/b08/nbafna1/data/wikimatrix/fr-oc/splits/fr/test", \
# "/export/b08/nbafna1/data/nllb/hin-bho/splits/hin/test", \
# "/export/b08/nbafna1/data/cvit-pib-v1.3/hi-mr/splits/hi/test", \
# "/export/b08/nbafna1/data/wikimatrix/fr-de/splits/fr/test", \
# "/export/b08/nbafna1/data/wikimatrix/es-en/splits/es/test", \
# ]

# DATAFILE_L2S=["/export/b08/nbafna1/data/wikimatrix/es-ca/splits/ca/test", \
# "/export/b08/nbafna1/data/wikimatrix/fr-oc/splits/oc/test", \
# "/export/b08/nbafna1/data/nllb/hin-bho/splits/bho/test", \
# "/export/b08/nbafna1/data/cvit-pib-v1.3/hi-mr/splits/mr/test", \
# "/export/b08/nbafna1/data/wikimatrix/fr-de/splits/de/test", \
# "/export/b08/nbafna1/data/wikimatrix/es-en/splits/en/test", \
# ]
LOGDIR = "/export/b08/nbafna1/projects/pgns-for-lrmt/output_analysis/logs"
# inputs.txt
DATAFILE_L1S=[f"{LOGDIR}/pgn{MODEL_NAMES[i]}/inputs.txt" for i in range(6)]
DATAFILE_L2S=[f"{LOGDIR}/pgn{MODEL_NAMES[i]}/references.txt" for i in range(6)]

lang_pairs = ["es-ca", "fr-oc", "hi-bh", "hi-mr", "fr-de", "es-en"]

OUTPUT_DIR = "test_set_splits_sentidx/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_DIR_FILES = "test_set_splits_files/"
os.makedirs(OUTPUT_DIR_FILES, exist_ok=True)

MODEL_OUTPUT_FILES = "model_output_files/"
os.makedirs(MODEL_OUTPUT_FILES, exist_ok=True)

for i in range(len(lang_pairs)):
    print(f"Processing {lang_pairs[i]}")
    # Get source and target sentences
    with open(DATAFILE_L1S[i], "r") as f:
        source_sents = f.readlines()
    with open(DATAFILE_L2S[i], "r") as f:
        target_sents = f.readlines()
    # Rank source sentences by shared tokens between source and target
        # normalized by min(len(source), len(target))

    print(f"Examples:")
    for j in random.sample(range(len(source_sents)), 5):
        print(f"Source: {source_sents[j].strip()}")
        print(f"Target: {target_sents[j].strip()}")

    sent2shared_token_percent = {}

    for j in range(len(source_sents)):
        source_sent = source_sents[j].strip().split()
        target_sent = target_sents[j].strip().split()
        shared_tokens = set(source_sent).intersection(set(target_sent))
        shared_token_percent = len(shared_tokens)/min(len(source_sent), len(target_sent))
        sent2shared_token_percent[j] = shared_token_percent

    # Sort source sentences by shared token percent
    sorted_sent2shared_token_percent = sorted(sent2shared_token_percent.items(), key=lambda x: x[1], reverse=True)

    # Take top 500 sentence idx
    top_500_sents = [sent_idx for sent_idx, _ in sorted_sent2shared_token_percent[:500]]
    print(f"Top 500 sents: {top_500_sents[:10]}")

    # From bottom 1000, sample 500 sentence idx
    # bottom_1000_sents = [sent_idx for sent_idx, _ in sorted_sent2shared_token_percent[-1000:]]
    # random.shuffle(bottom_1000_sents)
    # bottom_500_sents = bottom_1000_sents[:500]
    bottom_500_sents = [sent_idx for sent_idx, _ in sorted_sent2shared_token_percent[-500:]]
    print(f"Bottom 500 sents: {bottom_500_sents[:10]}")

    OUTPUT_DIR_LANG = f"{OUTPUT_DIR}/{lang_pairs[i]}/"
    os.makedirs(OUTPUT_DIR_LANG, exist_ok=True)

    with open(f"{OUTPUT_DIR_LANG}/top_500_sents.txt", "w") as f:
        for sent_idx in top_500_sents:
            f.write(str(sent_idx)+"\n")
    with open(f"{OUTPUT_DIR_LANG}/bottom_500_sents.txt", "w") as f:
        for sent_idx in bottom_500_sents:
            f.write(str(sent_idx)+"\n")
    
    OUTPUT_DIR_LANG_FILES = f"{OUTPUT_DIR_FILES}/{lang_pairs[i]}/"
    os.makedirs(OUTPUT_DIR_LANG_FILES, exist_ok=True)

    with open(f"{OUTPUT_DIR_LANG_FILES}/source_top_500_sents.txt", "w") as f:
        for sent_idx in top_500_sents:
            f.write(f"{source_sents[sent_idx].strip()}\n")
    with open(f"{OUTPUT_DIR_LANG_FILES}/target_top_500_sents.txt", "w") as f:
        for sent_idx in top_500_sents:
            f.write(f"{target_sents[sent_idx].strip()}\n")

    with open(f"{OUTPUT_DIR_LANG_FILES}/source_bottom_500_sents.txt", "w") as f:
        for sent_idx in bottom_500_sents:
            f.write(f"{source_sents[sent_idx].strip()}\n")
    with open(f"{OUTPUT_DIR_LANG_FILES}/target_bottom_500_sents.txt", "w") as f:
        for sent_idx in bottom_500_sents:
            f.write(f"{target_sents[sent_idx].strip()}\n")

    # Getting corresponding model output files
    model_names = {PGN_MODEL_NAMES[i], VANILLA_MODEL_NAMES[i]}
    for model_name in model_names:
        
        output_file = f"logs/{model_name}/predictions.txt"
        with open(output_file, "r") as f:
            model_output = f.readlines()

        MODEL_OUTPUT_DIR_LANG = f"{MODEL_OUTPUT_FILES}/{lang_pairs[i]}/{model_name}/"
        os.makedirs(MODEL_OUTPUT_DIR_LANG, exist_ok=True)

        with open(f"{MODEL_OUTPUT_DIR_LANG}/top_500_sents.txt", "w") as f:
            for sent_idx in top_500_sents:
                f.write(model_output[sent_idx].strip()+"\n")
        with open(f"{MODEL_OUTPUT_DIR_LANG}/bottom_500_sents.txt", "w") as f:
            for sent_idx in bottom_500_sents:
                f.write(model_output[sent_idx].strip()+"\n")

    



