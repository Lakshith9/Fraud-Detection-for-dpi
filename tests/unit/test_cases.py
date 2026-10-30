from api.routes.cases import refreshed_title


def test_default_title_follows_the_alert_count():
    assert refreshed_title("Investigation of 0 alert(s)", 1) == "Investigation of 1 alert(s)"
    assert refreshed_title("Investigation of 2 alert(s)", 5) == "Investigation of 5 alert(s)"


def test_unchanged_or_custom_titles_are_left_alone():
    assert refreshed_title("Investigation of 3 alert(s)", 3) is None
    assert refreshed_title("it case", 4) is None
    assert refreshed_title("Investigation of 2 alert(s) - mule ring", 4) is None
    assert refreshed_title("Ring R-6726e78e9d", 4) is None
    assert refreshed_title("", 1) is None
