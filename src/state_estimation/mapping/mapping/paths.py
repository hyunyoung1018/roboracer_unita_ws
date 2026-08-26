"""
Path helpers for the mapping stage.

Mapping is the one stage that creates a map folder rather than reading one, so
it needs the source `maps/` directory before the folder it will write exists.
That is a different problem from `sector_tuner/paths.py`, which resolves a map
folder that is already there.
"""

import os


def resolve_maps_source_dir(maps_dir: str) -> str:
    """
    Map an installed `share/stack_master/maps` path back to its source directory.

    `colcon build --symlink-install` symlinks *files*, not directories: every
    map folder under the install tree is a real directory whose contents are
    symlinks into src. So the usual trick of resolving the given directory's own
    entries does not work here - one level down they are all real directories.

    Resolving a file one level deeper does work, and any existing map will do:
    `<install>/maps/<some_map>/<some_map>.yaml` resolves to
    `<src>/maps/<some_map>/<some_map>.yaml`, whose grandparent is the source
    `maps/`. A new map folder is then created there, next to the existing ones.

    Falls back to the path as given when nothing resolvable is found - a plain
    (non-symlink) install, a path that already points into src, or a workspace
    with no maps yet. In that last case a map written into the install tree is
    lost on the next clean build, so the caller warns.
    """
    if not maps_dir or not os.path.isdir(maps_dir):
        return maps_dir

    for name in sorted(os.listdir(maps_dir)):
        entry = os.path.join(maps_dir, name)
        if os.path.islink(entry):
            return os.path.dirname(os.path.realpath(entry))
        if not os.path.isdir(entry):
            continue
        for inner in sorted(os.listdir(entry)):
            inner_path = os.path.join(entry, inner)
            if os.path.islink(inner_path):
                # <src>/maps/<map>/<file>  ->  <src>/maps
                return os.path.dirname(os.path.dirname(os.path.realpath(inner_path)))

    return maps_dir


def is_inside_install_tree(path: str) -> bool:
    """
    True when `path` sits under a colcon install tree.

    Used only to decide whether to warn: anything written there disappears on
    the next `rm -rf install`, which is not obvious at the moment of saving.
    """
    parts = os.path.abspath(path).split(os.sep)
    return 'install' in parts
