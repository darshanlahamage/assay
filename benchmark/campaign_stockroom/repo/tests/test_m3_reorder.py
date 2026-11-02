from stockroom import reorder_list


def test_below_threshold():
    assert reorder_list({"A": 2}, {"A": 10}, {"A": 4}) == [("A", 2)]


def test_at_threshold_no_order():
    assert reorder_list({"A": 10}, {"A": 10}, {"A": 4}) == []


def test_missing_level_counts_as_zero():
    assert reorder_list({}, {"B": 5}, {"B": 2}) == [("B", 3)]


def test_sorted_and_default_pack():
    assert reorder_list({"Z": 0, "M": 1}, {"Z": 2, "M": 3}, {}) == [("M", 2), ("Z", 2)]


def test_at_threshold_keeps_one_pack_spare():
    assert reorder_list({"C": 6}, {"C": 6}, {"C": 3}) == [("C", 1)]
