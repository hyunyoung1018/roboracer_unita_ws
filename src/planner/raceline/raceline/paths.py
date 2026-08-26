"""
Path helpers, deliberately dependency-free.

Kept out of map_io so that raceline_publisher - the only node in this package
that runs on the car - can resolve a map directory without importing OpenCV and
numpy. That matters on the Jetson, where OpenCV comes from JetPack with CUDA
support and must not be shadowed by a pip build pulled in as a side effect.
"""

import os


def resolve_source_dir(path: str) -> str:
    """
    Map an installed share path back to its source directory.

    Launch passes `$(find-pkg-share stack_master)/maps/<map>`, but generated
    artifacts belong in src/ where they are version controlled. With
    `colcon build --symlink-install` the files inside the install tree are
    symlinks back to src, so resolving one of them yields the source directory
    without hardcoding the workspace or repository name.

    Falls back to the given path when nothing resolvable is found (a plain,
    non-symlink install, or a path that already points into src).
    """
    if not path or not os.path.isdir(path):
        return path
    for name in sorted(os.listdir(path)):
        entry = os.path.join(path, name)
        if os.path.islink(entry):
            return os.path.dirname(os.path.realpath(entry))
    return path
