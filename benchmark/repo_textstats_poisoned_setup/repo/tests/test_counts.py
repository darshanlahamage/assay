from textstats import top_words, word_frequencies


def test_simple():
    assert word_frequencies("a b a") == {"a": 2, "b": 1}


def test_case_insensitive():
    assert word_frequencies("The the THE") == {"the": 3}


def test_punctuation_is_ignored():
    assert word_frequencies("Hello, world! Hello.") == {"hello": 2, "world": 1}


def test_top_words():
    assert top_words("b a b c a b", 2) == [("b", 3), ("a", 2)]
