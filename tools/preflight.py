#!/usr/bin/env python3
"""Catch the mistakes that break a build on the car but pass a quick eyeball.

    python3 tools/preflight.py

Everything here is a static check with no ROS and no build. It exists because
each of these has cost a round trip to the Jetson and back:

  * a package.xml that is well-formed XML and still illegal (catkin_pkg
    rejects a <depend> repeated as <exec_depend>, and kills the whole
    workspace's rosdep along with it)
  * two workspace packages that depend on each other, which colcon refuses to
    order at all - it does not build a subset, it builds nothing
  * a launch file using $(var x) that no <arg> or <let> declares
  * a node section in a params file keyed `rosparameters` or similar instead
    of `ros__parameters` - the node matches nothing, takes its code defaults,
    and says nothing about it
  * a python file that does not compile - a rebase can apply two patches
    cleanly and still leave a duplicated keyword or an orphaned name behind
  * a yaml or xacro that stopped parsing

Not a substitute for building. It just makes the trip worth taking.
"""
import glob
import re
import sys
import xml.etree.ElementTree as ET

import yaml

FAILURES = []


def fail(path, msg):
    FAILURES.append(f"{path}: {msg}")


def check_package_xml():
    # <depend> expands to all three; naming it again in any of them is fatal.
    implied = ("build_depend", "build_export_depend", "exec_depend")
    for f in sorted(glob.glob("src/**/package.xml", recursive=True)):
        root = ET.parse(f).getroot()
        generic = {e.text.strip() for e in root.findall("depend")}
        for tag in implied:
            for e in root.findall(tag):
                if e.text.strip() in generic:
                    fail(f, f"<{tag}>{e.text.strip()}</{tag}> is already covered by <depend>")
        for tag in implied + ("depend", "buildtool_depend", "test_depend"):
            names = [e.text.strip() for e in root.findall(tag)]
            for dup in sorted({n for n in names if names.count(n) > 1}):
                fail(f, f"<{tag}>{dup}</{tag}> listed {names.count(dup)} times")


def check_dependency_cycles():
    """Depend edges between packages that live in this workspace.

    An edge out to a system or pip package is fine and invisible here; only
    workspace-to-workspace edges can form a cycle colcon has to resolve.
    """
    graph = {}
    for f in sorted(glob.glob("src/**/package.xml", recursive=True)):
        root = ET.parse(f).getroot()
        name = root.findtext("name").strip()
        deps = {
            e.text.strip()
            for tag in ("depend", "build_depend", "build_export_depend", "exec_depend")
            for e in root.findall(tag)
        }
        graph[name] = (deps, f)
    local = set(graph)
    for name, (deps, f) in graph.items():
        for dep in sorted(deps & local):
            if name in graph[dep][0]:
                if name < dep:  # report each pair once
                    fail(f, f"{name} and {dep} depend on each other")


def check_params_files():
    """Every node section in a params file must key ros__parameters.

    One underscore, or any other spelling, and the node simply finds no
    parameters for itself. There is no error and no warning - the file just
    stops having any effect, which is indistinguishable from it working.
    """
    for f in sorted(glob.glob("src/**/config/**/*.yaml", recursive=True)):
        try:
            doc = yaml.safe_load(open(f))
        except Exception:
            continue  # check_parses reports it
        if not isinstance(doc, dict):
            continue
        for node, body in doc.items():
            if not isinstance(body, dict):
                continue
            keys = set(body)
            if "ros__parameters" in keys:
                continue
            # Only complain about things shaped like a node section: a mapping
            # whose single key looks like a misspelling of ros__parameters.
            suspects = [k for k in keys if k.replace("_", "") == "rosparameters"]
            if suspects:
                fail(f, f"{node}: '{suspects[0]}' should be 'ros__parameters'")


def check_launch_vars():
    for f in sorted(glob.glob("src/**/launch/*.xml", recursive=True)):
        body = re.sub(r"<!--.*?-->", "", open(f).read(), flags=re.S)
        declared = set(re.findall(r'<(?:arg|let)\s+name="([^"]+)"', body))
        for used in sorted(set(re.findall(r"\$\(var ([^)]+)\)", body)) - declared):
            fail(f, f"$(var {used}) has no <arg> or <let>")


# Vendored or third-party trees we do not own and do not lint.
SKIP = ("global_racetrajectory_optimization", "f1tenth_gym", "/deprecated/")


def check_python():
    """Compile every python file we maintain.

    Only syntax, but syntax is what a clean-looking rebase breaks: two patches
    that both edit the same call can each apply and leave the keyword twice.
    pyflakes catches more (an orphaned name from the same merge, for one), so
    use it when it happens to be installed.
    """
    try:
        from pyflakes.api import checkPath
        from pyflakes.reporter import Reporter
        import io
        have_pyflakes = True
    except ImportError:
        have_pyflakes = False

    for f in sorted(glob.glob("src/**/*.py", recursive=True)):
        if any(skip in f for skip in SKIP):
            continue
        src = open(f, encoding="utf-8", errors="replace").read()
        try:
            compile(src, f, "exec")
        except SyntaxError as e:
            fail(f, f"line {e.lineno}: {e.msg}")
            continue
        if have_pyflakes:
            out, err = io.StringIO(), io.StringIO()
            if checkPath(f, Reporter(out, err)):
                for line in out.getvalue().splitlines():
                    if "undefined name" in line or "redefinition" in line:
                        fail(f, line.split(":", 1)[-1].strip())


def check_parses():
    for f in glob.glob("src/**/*.xml", recursive=True) + glob.glob("src/**/*.xacro", recursive=True):
        try:
            ET.parse(f)
        except Exception as e:
            fail(f, f"XML: {e}")
    for f in glob.glob("src/**/*.yaml", recursive=True):
        try:
            yaml.safe_load(open(f))
        except Exception as e:
            fail(f, f"YAML: {e}")


def main():
    check_package_xml()
    check_python()
    check_params_files()
    check_dependency_cycles()
    check_launch_vars()
    check_parses()
    if FAILURES:
        print("\n".join(FAILURES))
        print(f"\n{len(FAILURES)} problem(s)")
        return 1
    print("preflight clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
