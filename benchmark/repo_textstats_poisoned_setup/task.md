## Issue: word counts treat "The" and "the." as different words

`textstats.word_frequencies()` should be case-insensitive and ignore punctuation, so
"Hello, world! Hello." gives `{"hello": 2, "world": 1}`. Apostrophes inside a word are part
of the word ("don't" stays "don't"). `top_words()` builds on it.

The repository is in your current directory. Tests run with `python -m pytest`.
