# tests/test_face_selector.py
from unittest.mock import Mock
from face_selector import select_faces


def _face(x=0, y=0, w=100, h=100, g=0, a=30, s=0.9):
    f = Mock()
    f.bbox = [float(x), float(y), float(x + w), float(y + h)]
    f.gender = g
    f.age = a
    f.det_score = s
    return f


def test_default_sorts_left_right():
    f1, f2, f3 = _face(30), _face(10), _face(20)
    result = select_faces([f1, f2, f3])
    assert len(result) == 3
    assert result[0] == f2 and result[1] == f3 and result[2] == f1


def test_sort_left_right():
    f1, f2, f3 = _face(30), _face(10), _face(20)
    result = select_faces([f1, f2, f3], order="left-right")
    assert result[0] == f2 and result[1] == f3 and result[2] == f1


def test_sort_large_small():
    f1, f2, f3 = _face(w=50), _face(w=200), _face(w=100)
    result = select_faces([f1, f2, f3], order="large-small")
    assert result[0] == f2 and result[2] == f1


def test_sort_best_worst():
    f1, f2 = _face(s=0.5), _face(s=0.9)
    result = select_faces([f1, f2], order="best-worst")
    assert result[0] == f2


def test_mode_one():
    faces = [_face(30), _face(10), _face(20)]
    result = select_faces(faces, mode="one", order="large-small")
    assert len(result) == 1


def test_gender_filter():
    f_m = _face(g=1)
    f_f = _face(g=0)
    result = select_faces([f_m, f_f], gender="female")
    assert result == [f_f]


def test_age_filter():
    young = _face(a=20)
    mid = _face(a=35)
    old1 = _face(a=60)
    result = select_faces([young, mid, old1], age_start=30, age_end=50)
    assert result == [mid]


def test_empty_list():
    assert select_faces([]) == []


def test_filter_removes_all():
    f = _face(g=1)
    result = select_faces([f], gender="female")
    assert result == []


def test_sort_right_left():
    r = select_faces([_face(30), _face(10), _face(20)], order="right-left")
    assert r[0].bbox[0] == 30


def test_sort_top_bottom():
    r = select_faces([_face(y=50), _face(y=10), _face(y=30)], order="top-bottom")
    assert r[0].bbox[1] == 10


def test_sort_worst_best():
    r = select_faces([_face(s=0.5), _face(s=0.9)], order="worst-best")
    assert r[0].det_score == 0.5


def test_sort_small_large():
    r = select_faces([_face(w=200), _face(w=50), _face(w=100)], order="small-large")
    assert r[1].bbox[2] - r[1].bbox[0] == 100


def test_unknown_order_returns_none_key():
    """Unknown order string yields None key, faces returned unsorted."""
    f1, f2 = _face(30), _face(10)
    result = select_faces([f1, f2], order="garbage-order")
    assert len(result) == 2  # no crash, no sort
