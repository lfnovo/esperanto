#!/usr/bin/env python3
"""Build and validate Esperanto distributions outside the source checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "esperanto"
RUNTIME_FILES = {
    path.relative_to(ROOT / "src").as_posix()
    for path in (ROOT / "src" / PACKAGE).rglob("*.py")
}
RUNTIME_FILES.add(f"{PACKAGE}/py.typed")


def run(command: list[str], *, cwd: Path) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def project_version() -> str:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    if match is None:
        raise RuntimeError("could not read project version from pyproject.toml")

    version = match.group(1)
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    lock_match = re.search(
        r'^name = "esperanto"\nversion = "([^"]+)"$', lock, re.MULTILINE
    )
    if lock_match is None or lock_match.group(1) != version:
        locked = lock_match.group(1) if lock_match else "missing"
        raise RuntimeError(
            f"version files disagree: pyproject.toml={version}, uv.lock={locked}"
        )
    return version


def one_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern} in {directory}, found {len(matches)}"
        )
    return matches[0].resolve()


def wheel_metadata(wheel: Path) -> tuple[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError("wheel must contain exactly one METADATA file")
        metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))
        return metadata.get("Name", ""), metadata.get("Version", "")


def inspect_wheel(wheel: Path, version: str) -> None:
    name, built_version = wheel_metadata(wheel)
    if name != PACKAGE or built_version != version:
        raise RuntimeError(
            f"wheel metadata is {name} {built_version}, expected {PACKAGE} {version}"
        )

    with zipfile.ZipFile(wheel) as archive:
        packaged = set(archive.namelist())
    missing = sorted(RUNTIME_FILES - packaged)
    if missing:
        raise RuntimeError(f"wheel is missing runtime files: {', '.join(missing)}")


def inspect_sdist(sdist: Path, version: str) -> None:
    expected_prefix = f"{PACKAGE}-{version}/"
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
        if not all(
            name == expected_prefix[:-1] or name.startswith(expected_prefix)
            for name in names
        ):
            raise RuntimeError("sdist contains entries outside its versioned root")
        packaged = {
            name.removeprefix(expected_prefix)
            for name in names
            if name.startswith(expected_prefix)
        }
        metadata_names = [name for name in names if name.endswith("/PKG-INFO")]
        if len(metadata_names) != 1:
            raise RuntimeError("sdist must contain exactly one PKG-INFO file")
        metadata_file = archive.extractfile(metadata_names[0])
        if metadata_file is None:
            raise RuntimeError("could not read sdist PKG-INFO")
        metadata = metadata_file.read().decode("utf-8")

    missing = sorted({f"src/{path}" for path in RUNTIME_FILES} - packaged)
    if missing:
        raise RuntimeError(f"sdist is missing runtime files: {', '.join(missing)}")
    if f"Name: {PACKAGE}\n" not in metadata or f"Version: {version}\n" not in metadata:
        raise RuntimeError("sdist metadata does not match project name and version")


def smoke_wheel(wheel: Path, version: str, workdir: Path) -> None:
    code = """
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path

import esperanto
from esperanto import AIFactory

origin = Path(esperanto.__file__).resolve()
checkout = Path(os.environ["ESPERANTO_CHECKOUT_ROOT"]).resolve()
if origin == checkout or checkout in origin.parents:
    raise RuntimeError(f"import resolved inside checkout: {origin}")
if importlib.metadata.version("esperanto") != os.environ["ESPERANTO_EXPECTED_VERSION"]:
    raise RuntimeError("installed metadata version does not match candidate")
if importlib.util.find_spec("transformers") is not None:
    raise RuntimeError("bare wheel unexpectedly installed the transformers extra")
if importlib.util.find_spec("jsonschema") is not None:
    raise RuntimeError("bare wheel unexpectedly installed the validation extra")
providers = AIFactory.get_available_providers()
if not providers.get("language") or not providers.get("text_to_speech"):
    raise RuntimeError("credential-free factory discovery returned incomplete surfaces")
print(json.dumps({"module_origin": str(origin), "version": importlib.metadata.version("esperanto")}))
"""
    env = os.environ.copy()
    env["ESPERANTO_CHECKOUT_ROOT"] = str(ROOT)
    env["ESPERANTO_EXPECTED_VERSION"] = version
    command = [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--with",
        str(wheel),
        "python",
        "-I",
        "-c",
        code,
    ]
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=workdir, env=env, check=True)


def check_extra(
    wheel: Path, extra: str, distributions: list[str], workdir: Path
) -> None:
    quoted = json.dumps(distributions)
    code = (
        "import importlib.metadata; "
        f"names = {quoted}; "
        "print({name: importlib.metadata.version(name) for name in names})"
    )
    run(
        [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--with",
            f"{PACKAGE}[{extra}] @ {wheel.as_uri()}",
            "python",
            "-I",
            "-c",
            code,
        ],
        cwd=workdir,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    version = project_version()
    with tempfile.TemporaryDirectory(prefix="esperanto-package-check-") as temporary:
        workspace = Path(temporary)
        dist = workspace / "dist"
        rebuilt = workspace / "rebuilt"
        clean_room = workspace / "clean-room"
        dist.mkdir()
        rebuilt.mkdir()
        clean_room.mkdir()

        run(["uv", "build", "--out-dir", str(dist), str(ROOT)], cwd=workspace)
        wheel = one_file(dist, "*.whl")
        sdist = one_file(dist, "*.tar.gz")
        inspect_wheel(wheel, version)
        inspect_sdist(sdist, version)
        smoke_wheel(wheel, version, clean_room)

        run(
            ["uv", "build", "--wheel", "--out-dir", str(rebuilt), str(sdist)],
            cwd=workspace,
        )
        rebuilt_wheel = one_file(rebuilt, "*.whl")
        inspect_wheel(rebuilt_wheel, version)
        smoke_wheel(rebuilt_wheel, version, clean_room)

        check_extra(
            wheel,
            "transformers",
            ["transformers", "torch", "sentence-transformers"],
            clean_room,
        )
        check_extra(wheel, "validation", ["jsonschema"], clean_room)

        result = {
            "package": PACKAGE,
            "version": version,
            "python": sys.version.split()[0],
            "artifacts": {
                wheel.name: sha256(wheel),
                sdist.name: sha256(sdist),
                f"sdist-rebuilt/{rebuilt_wheel.name}": sha256(rebuilt_wheel),
            },
        }
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
