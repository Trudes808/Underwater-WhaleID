import datetime as dt

from uwid import names


def test_first_tag_name():
    i = names.parse_tag_video_name("/x/2025-2-3 Mini-mother 11-96/Video/CC-11-96-20250203-101236-000-00004.mov")
    assert i == {"tag": "CC-11-96", "extra": None, "start_local": dt.datetime(2025, 2, 3, 10, 12, 36), "field": "000",
                 "seq": 4}


def test_name_with_extra_part():
    i = names.parse_tag_video_name("CC-09-85-HRA-20250102-111224-000-00005.mov")
    assert i["tag"] == "CC-09-85" and i["extra"] == "HRA" and i["seq"] == 5 and i["field"] == "000"
    assert i["start_local"] == dt.datetime(2025, 1, 2, 11, 12, 24)
    assert names.date_hint("CC-09-85-HRA-20250102-111224-000-00005.mov") == dt.date(2025, 1, 2)
    assert names.start_seconds_of_day("CC-09-85-HRA-20250102-111224-000-00023.mov") == 11 * 3600 + 12 * 60 + 24


def test_bad_names():
    assert names.parse_tag_video_name("2025-12-20 CG.m4v") is None
    assert names.parse_tag_video_name("CC-09-85-HRA-20251399-111224-000-00005.mov") is None
    assert names.date_hint("/footage/2025-12-20 CG.m4v") == dt.date(2025, 12, 20)
    assert names.start_seconds_of_day("clip.mov") is None


def test_sensor_names():
    a = names.parse_sensor_name("20250102-225123-CC-09-85-HRA.csv")
    assert a == {"tag": "CC-09-85", "extra": "HRA", "written": dt.datetime(2025, 1, 2, 22, 51, 23)}
    b = names.parse_sensor_name("20250203-164014-CC-11-96.csv")
    assert b["tag"] == "CC-11-96" and b["extra"] is None
