"""Policy chooses among SAFE sides; it never creates one.

_available_sides has already refused any side that does not clear every
obstacle by spline_bound_mindist. _policy_side only decides which of the
survivors to take, so a preference for an unsafe side is ignored and the car
goes the other way rather than not at all.
"""

from types import SimpleNamespace

import importlib

module = importlib.import_module('lane_change_planner.change_avoidance_node')
Node = next(
    obj for obj in vars(module).values()
    if isinstance(obj, type) and hasattr(obj, '_policy_side'))


def planner(preferred='auto'):
    p = Node.__new__(Node)
    p.preferred_side = preferred
    p.get_logger = lambda: SimpleNamespace(info=lambda *a, **k: None)
    return p


def rooms(left=None, right=None):
    """Room per side, as _select_side builds it: {side: [(id, room), ...]}."""
    out = {}
    if left is not None:
        out['left'] = [(1, left)]
    if right is not None:
        out['right'] = [(1, right)]
    return out


def choose(preferred, **kw):
    return Node._policy_side(planner(preferred), rooms(**kw))


def test_auto_is_the_room_comparison_exactly():
    assert choose('auto', left=0.10, right=0.30) == 'right'
    assert choose('auto', left=0.30, right=0.10) == 'left'


def test_a_named_side_is_taken_even_with_less_room():
    """The whole point: the inside is shorter, and room cannot see that."""
    assert choose('left', left=0.10, right=0.30) == 'left'


def test_a_named_side_that_is_not_safe_is_ignored():
    """_available_sides already dropped it, so it is not in the dict."""
    assert choose('left', right=0.30) == 'right'


def test_a_named_side_that_is_also_the_roomier_one_is_still_taken():
    assert choose('right', left=0.10, right=0.30) == 'right'


def test_one_safe_side_is_taken_whatever_the_policy_says():
    assert choose('auto', left=0.05) == 'left'
    assert choose('right', left=0.05) == 'left'


def test_the_tightest_point_decides_under_auto():
    p = planner('auto')
    # left is roomier at one obstacle but tighter at another; min() decides.
    available = {'left': [(1, 0.40), (2, 0.05)], 'right': [(1, 0.20), (2, 0.20)]}
    assert Node._policy_side(p, available) == 'right'
