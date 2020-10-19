# gendered-analogies
Word analogy task performed on male and female words for professions in Slovene language

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
