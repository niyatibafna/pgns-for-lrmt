# How many tokens could be copied from the source to the target at most?

# Doing for Spanish-Catalan
# file1 = "/export/b08/nbafna1/data/europarl.es-ca/splits/es/" 
# file2 = "/export/b08/nbafna1/data/europarl.es-ca/splits/ca/" 

# file1 = "/export/b08/nbafna1/data/wikimatrix/es-ca/splits/es/"
# file2 = "/export/b08/nbafna1/data/wikimatrix/es-ca/splits/ca/"

# file1 = "/export/b08/nbafna1/data/cvit-pib-v1.3/hi-mr/splits/hi/"
# file2 = "/export/b08/nbafna1/data/cvit-pib-v1.3/hi-mr/splits/mr/"

# file1 = "/export/b08/nbafna1/data/nllb/hin-bho/splits/hin"
# file2 = "/export/b08/nbafna1/data/nllb/hin-bho/splits/bho"

# file1 = "/export/b08/nbafna1/data/wikimatrix/fr-oc/splits/fr/"
# file2 = "/export/b08/nbafna1/data/wikimatrix/fr-oc/splits/oc/" 

# file1 = "/export/b08/nbafna1/data/wikimatrix/es-ca/splits/es/"
# file2 = "/export/b08/nbafna1/data/wikimatrix/es-ca/splits/ca/"

# file1 = "/export/b08/nbafna1/data/wikimatrix/fr-de/splits/fr/"
# file2 = "/export/b08/nbafna1/data/wikimatrix/fr-de/splits/de/" 

file1 = "/export/b08/nbafna1/data/wikimatrix/es-en/splits/es/"
file2 = "/export/b08/nbafna1/data/wikimatrix/es-en/splits/en/" 

file1 = file1 + "/train"
file2 = file2 + "/train"

common_tokens = 0
lines = 0
total_target_tokens = 0
src_sentence_length = 0
tgt_sentence_length = 0

with open(file1, "r") as f1, open(file2, "r") as f2:
    for line1, line2 in zip(f1, f2):
        tokens1 = line1.split()
        tokens2 = line2.split()
        common_tokens += len(set(tokens1).intersection(set(tokens2)))
        lines += 1
        total_target_tokens += len(tokens2)
        src_sentence_length += len(tokens1)
        tgt_sentence_length += len(tokens2)


print("Common tokens: ", common_tokens)
print("Total target tokens: ", total_target_tokens)
print("Total lines: ", lines)
print("Average common tokens per line: ", common_tokens / lines)
print("Average common tokens per target token: ", common_tokens / total_target_tokens)
print("Average source sentence length: ", src_sentence_length / lines)
print("Average target sentence length: ", tgt_sentence_length / lines)

