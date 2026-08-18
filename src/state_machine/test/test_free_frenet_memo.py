"""The per-tick memo must never serve a stale verdict.

Covers H2HStateMachine._memo_free_frenet. One loop asks the same cache
the same question up to three times, and the answer cannot change in between -
but only while the tick and the cache generation are the same, which is what
the key checks. Lives in the wrapper because state_machine_node.py is what
time_trials runs.
"""

from types import SimpleNamespace

import state_machine.h2h_state_machine as module
from state_machine.h2h_state_machine import H2HStateMachine


def harness(answers):
    """A real instance, uninitialised, whose shared check returns `answers`.

    `__new__` rather than a SimpleNamespace because _memo_free_frenet uses the
    zero-argument `super()`, which needs `self` to be an instance of the class.
    """
    calls = []

    def shared(wpnts_data):
        calls.append(wpnts_data)
        return answers[len(calls) - 1]

    state = H2HStateMachine.__new__(H2HStateMachine)
    state._loop_seq = 1
    state._free_memo = {}
    state._shared_free = shared
    cache = SimpleNamespace(name='static_avoidance_planner',
                            init_count=0, is_init=True)
    return state, cache, calls


def ask(state, cache):
    """Call the memo with the shared _check_free_frenet stubbed out."""
    original = module.StateMachine._check_free_frenet
    module.StateMachine._check_free_frenet = (
        lambda self, wpnts_data: self._shared_free(wpnts_data))
    try:
        return H2HStateMachine._memo_free_frenet(state, cache)
    finally:
        module.StateMachine._check_free_frenet = original


def test_second_call_in_a_tick_is_served_from_the_memo():
    state, cache, calls = harness([False, True])
    assert ask(state, cache) is False
    assert ask(state, cache) is False
    assert len(calls) == 1


def test_next_tick_recomputes():
    state, cache, calls = harness([False, True])
    assert ask(state, cache) is False
    state._loop_seq += 1
    assert ask(state, cache) is True
    assert len(calls) == 2


def test_mid_loop_cache_refresh_invalidates():
    # _check_latest_wpnts -> initialize_traj bumps init_count without the tick
    # changing, and several callers run it right before asking.
    state, cache, calls = harness([False, True])
    assert ask(state, cache) is False
    cache.init_count += 1
    assert ask(state, cache) is True
    assert len(calls) == 2


def test_losing_the_path_invalidates():
    # _expire_stale_cache and the source-change rule both clear is_init, which
    # flips what the check returns for an OT cache.
    state, cache, calls = harness([True, False])
    assert ask(state, cache) is True
    cache.is_init = False
    assert ask(state, cache) is False
    assert len(calls) == 2


def test_two_caches_do_not_share_an_entry():
    state, first, calls = harness([False, True])
    second = SimpleNamespace(name='global_tracking', init_count=0, is_init=True)
    assert ask(state, first) is False
    assert ask(state, second) is True
    assert len(calls) == 2
