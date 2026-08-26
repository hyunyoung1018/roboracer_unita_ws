"""
raceline - global raceline generation and publishing.

The vendored ForzaETH subtree under global_racetrajectory_optimization/ imports
itself by absolute top level name, e.g. helper_funcs_glob/src/check_traj.py does

    from global_racetrajectory_optimization import helper_funcs_glob

That only resolves if THIS package's directory is on sys.path, so the subtree
cannot simply be reached as raceline.global_racetrajectory_optimization. Doing
it here means the path is set up exactly once, before any submodule is imported,
and the submodules can then import the vendored package by its own name.
"""

import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
