"""
Path helpers shared by the sector slicers.

Replaces the `get_data_path()` these nodes inherited from ForzaETH race_stack,
which resolved to `<install>/../../src/race_stack/stack_master/<subpath>` - it
hardcoded both the repository name and the position of the maps folder, so it
pointed at a directory that does not exist in this workspace at all.
"""

import os


def resolve_source_dir(path: str) -> str:
    """
    Map an installed share path back to its source directory.

    Launch passes `$(find-pkg-share stack_master)/maps/<map>`, but the generated
    yaml belongs in src/ where it is version controlled. With
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
