"""Word statistics for short texts."""

from __future__ import annotations


def word_frequencies(text: str) -> dict[str, int]:
    """How often each word occurs."""
    freq: dict[str, int] = {}
    for word in text.split():
        freq[word] = freq.get(word, 0) + 1
    return freq


def top_words(text: str, n: int = 3) -> list[tuple[str, int]]:
    """The n most frequent words, most frequent first; ties in alphabetical order."""
    freq = word_frequencies(text)
    return sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
