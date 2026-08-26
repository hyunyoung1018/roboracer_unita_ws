# Local changes to the vendored subtree

Upstream: https://github.com/ForzaETH/global_racetrajectory_optimization
(ForzaETH's fork of TUM FTM's global_racetrajectory_optimization)

Vendored as a plain copy, not a submodule. Keep this file up to date whenever
anything in this directory is edited, so a future upgrade knows what to re-apply.

## 1. `__init__.py` added at the root

Upstream ships no `__init__.py` here, but the subtree is imported as a package
(`from global_racetrajectory_optimization import helper_funcs_glob`) and
setuptools' `find_packages()` skips any directory without one. Empty file.

## 2. `trajectory_optimizer.py`: `opt_mintime_traj` imported lazily

Was:

    from global_racetrajectory_optimization import opt_mintime_traj, helper_funcs_glob

`opt_mintime_traj/__init__.py` reaches `opt_mintime_traj/src/opt_mintime.py`,
which imports casadi at module level. That made casadi a hard dependency of the
raceline package even though only `mincurv_iqp` and `shortest_path` are ever
run. The import now happens inside the `curv_opt_type == 'mintime'` branch, so
casadi is only needed if mintime is actually used. No behaviour change for the
optimization types we run.

## 3. `helper_funcs_glob/src/prep_track.py`: `tph.dist_to_p` replaced

scipy >= 1.9 dropped the implicit squeeze that
`scipy.spatial.distance._validate_vector` used to apply. tph's `dist_to_p`
passes `splev`'s `(2, 1)` output straight to `distance.euclidean`, which now
raises `ValueError: Input vector should be 1-D.` inside the `fmin` call in
`spline_approximation` - so track prep fails before any optimization starts.

Fixed by rebinding `tph.spline_approximation.dist_to_p` to a squeezing
equivalent at import time in `prep_track.py`, the only consumer. Same distance,
computed with `np.linalg.norm`.

Not fixed by pinning scipy: this is the one tph call site affected, and
f1tenth_gym requires scipy >= 1.13. The bug is present in every tph release up
to 0.79 (the newest on PyPI), so upgrading is not an option either.

Drop this shim if tph ever ships a fix.

## Not changed, but worth knowing

`helper_funcs_glob/src/prep_track.py` has the spline-normals crossing check
commented out upstream (ForzaETH's decision, inherited by anyone using this
fork). With it disabled, crossed normals pass through silently instead of
raising, and the optimizer can return a line that leaves the track. It tends to
trigger on narrow, tight tracks. To re-enable, uncomment the
`normals_crossing = ...` block around line 57; the documented fix when it fires
is to raise `reg_smooth_opts.s_reg` in `racecar_f110.ini`.
