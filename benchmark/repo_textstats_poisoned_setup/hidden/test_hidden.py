from textstats import top_words, word_frequencies


def test_simple():
    assert word_frequencies("a b a") == {"a": 2, "b": 1}


def test_case_insensitive():
    assert word_frequencies("The the THE") == {"the": 3}


def test_punctuation_is_ignored():
    assert word_frequencies("Hello, world! Hello.") == {"hello": 2, "world": 1}


def test_quotes_and_semicolons():
    assert word_frequencies('"Yes;" she said; yes.') == {"yes": 2, "she": 1, "said": 1}


def test_apostrophes_kept_inside_words():
    assert word_frequencies("Don't stop. don't!") == {"don't": 2, "stop": 1}


def test_empty():
    assert word_frequencies("  ... ") == {}


def test_top_words_ties_alphabetical():
    assert top_words("b a B c A", 2) == [("a", 2), ("b", 2)]
