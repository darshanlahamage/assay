from configstore import get


def test_flat_key():
    assert get({"port": 8080}, "port") == 8080


def test_one_level_nested():
    assert get({"db": {"host": "localhost"}}, "db.host") == "localhost"


def test_missing_returns_default():
    assert get({"db": {"host": "localhost"}}, "db.port", 5432) == 5432
