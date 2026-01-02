from collections import Counter
import os
import re
import numpy as np
import pandas as pd
from nltk import WordPunctTokenizer

import core.vars as vars

def text_preprocessor(texts: pd.Series, min_freq: int=5) -> tuple[pd.Series, dict[str, int]]:
    tokenizer = WordPunctTokenizer()

    tokens_counts = Counter()
    vocabulary = {}

    tokens = texts.copy()

    for i in range(len(tokens)):
        text = tokens.iloc[i]
        # text = re.sub(r"\W", " ", text)
        # text = re.sub(r"\s+", " ", text)
        # texts[i] = text

        text = re.sub(r'[.,´’!?;:"\'()\[\]{}<>«»„“”\-–—/\\|@#$%^&*_+=~→`]', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)  # Множественные пробелы/табы → один пробел
        text = re.sub(r'\n[ \t]+\n', '\n\n', text)  # Убираем пробелы между переносами
        text = text.lower()

        words = tokenizer.tokenize(text)
        tokens_counts.update(words)

        tokens.iloc[i] = text.strip()

    idx = 0
    for word, count in tokens_counts.most_common():
        if count >= min_freq and word not in vocabulary:
            vocabulary[word] = idx
            idx += 1
        else:
            break

    vocabulary[vars.UNK_VAL] = idx

    return tokens, vocabulary
