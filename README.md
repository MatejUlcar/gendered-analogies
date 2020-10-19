# gendered-analogies
Word analogy task performed on male and female words for professions in Slovene language

Results published in:

Anka Supej, Matej Ulčar, Marko Robnik Šikonja, Senja Pollak: \
Dimenzija spola v slovenskih vektorskih vložitvah besed: primerjava modelov prek analogij poklicev \
Conference: Jezikovne tehnologije in digitalna humanistika 2020, Ljubljana, 2020

## Requirements
numpy
lemmagen3

## Usage
Run `run_analogies.py` with `--help` to see all the options, an example below:
```
python run_analogies.py \
  --embeddings ../fasttext/sketchengine_fasttext/sltenten15.vec \
  --lemmatize \
  -n 50 \
  --output results/ft_sketchengine_lem_avg \
  --avginput
```
Embeddings are expected to be in text format, with each word in its own row. Each row should have a word/token followed by the word vector, where each vector element is space separated and vector is space separated from the token. The first line in embeddings file should have two space separated numbers, the first representing the number of tokens in the file, the second representing the dimension of the vectors.

The input file (list of professions) should be in tsv format, where male profession words should be in the 3rd column, an optional synonym in the 5th column and optional frequency counters for profession and its synonym in 4th and 6th column, respectively. Female profession words should be in the 6th column and its synonym in 8th column (optionally). The optional frequency counters for female professions should be in 7th and 9th column. If, for example, there is no synonym, leave that column empty, do not skip it entirely.
