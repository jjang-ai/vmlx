#!/usr/bin/env python3
"""Prepare and attest the MLX wheel pair selected by mlx-runtime.json."""
from __future__ import annotations

import argparse
import email
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import zipfile

SPEC_PATH = Path(__file__).with_name("mlx-runtime.json")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def wheel_records(directory, spec, target):
    records = []
    for name in ("mlx", "mlx_metal"):
        matches = list(directory.glob(f"{name}-{spec['version']}-*{target}.whl"))
        if len(matches) != 1:
            raise ValueError(f"Expected one {name} wheel for {target}: {matches}")
        path = matches[0]
        with zipfile.ZipFile(path) as archive:
            metadata_names = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
            wheel_names = [n for n in archive.namelist() if n.endswith('.dist-info/WHEEL')]
            if len(metadata_names) != 1 or len(wheel_names) != 1:
                raise ValueError(f"Invalid wheel metadata: {path}")
            metadata = email.message_from_bytes(archive.read(metadata_names[0]))
            wheel = email.message_from_bytes(archive.read(wheel_names[0]))
            if metadata['Version'] != spec['version'] or metadata['Name'].replace('-', '_') != name:
                raise ValueError(f"Wheel identity differs from runtime specification: {path}")
            tags = wheel.get_all('Tag', [])
            if not tags or {t.rsplit('-', 1)[-1] for t in tags} != {target}:
                raise ValueError(f"Wheel platform differs from runtime specification: {path}")
        records.append({'filename': path.name, 'sha256': sha256(path), 'tags': tags})
    return records


def verify(directory, spec, target):
    receipt = json.loads((directory / 'mlx-wheel-provenance.json').read_text())
    expected = {
        'runtime_spec_sha256': sha256(SPEC_PATH),
        'version': spec['version'], 'source': spec['source'],
        'platform': target, 'target': spec['platforms'][target],
        'wheels': wheel_records(directory, spec, target),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ValueError(f"MLX wheel provenance mismatch: {key}")
    return receipt


def prepare(directory, spec, target, build_root):
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / 'mlx-wheel-provenance.json'
    if receipt_path.exists():
        verify(directory, spec, target)
        return
    source = spec['source']
    if source['kind'] == 'pypi':
        for name, impl, abi in [('mlx', 'cp', 'cp312'), ('mlx-metal', 'py', 'none')]:
            run([sys.executable, '-m', 'pip', 'download', '--only-binary=:all:', '--no-deps',
                 '--dest', directory, '--platform', target, '--implementation', impl,
                 '--python-version', '312', '--abi', abi, f"{name}=={spec['version']}"])
        build_info = {'kind': 'official-wheel'}
    elif source['kind'] == 'git':
        if platform.system() != 'Darwin' or platform.machine() != 'arm64' or sys.version_info[:2] != (3, 12):
            raise ValueError('MLX source wheels require macOS arm64 and Python 3.12')
        root = build_root / source['commit'] / target
        root.mkdir(parents=True, exist_ok=True)
        checkout = root / 'source'
        if not checkout.exists():
            run(['git', 'clone', '--filter=blob:none', '--no-checkout', source['repository'], checkout])
            run(['git', '-C', checkout, 'checkout', '--detach', source['commit']])
        head = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
        if head != source['commit']:
            raise ValueError(f'MLX checkout drift: {head}')
        # Only package-version metadata changes. All runtime/kernel sources remain upstream.
        setup = subprocess.check_output(['git', '-C', str(checkout), 'show', 'HEAD:setup.py'], text=True)
        marker = '    version = get_version()'
        if setup.count(marker) != 1:
            raise ValueError('Upstream setup.py changed; review the version-only overlay')
        patched = setup.replace(marker, f"    version = {spec['version']!r}")
        setup_path = checkout / 'setup.py'
        if setup_path.read_text() not in (setup, patched):
            raise ValueError('Unexpected MLX setup.py changes')
        changed = subprocess.check_output(['git', '-C', str(checkout), 'diff', '--name-only', 'HEAD'], text=True).splitlines()
        if set(changed) - {'setup.py'}:
            raise ValueError(f'Unexpected MLX source changes: {changed}')
        setup_path.write_text(patched)
        venv = root / 'build-venv'
        if not (venv / 'bin/python').exists():
            run([sys.executable, '-m', 'venv', venv])
        python = venv / 'bin/python'
        # Build tools have their own lock; the application's runtime constraints
        # must not silently change the compiler/package toolchain.
        build_env = {k: v for k, v in os.environ.items() if k not in {'PIP_CONSTRAINT', 'PIP_BUILD_CONSTRAINT'}}
        run([python, '-m', 'pip', 'install', *spec['build_requirements']], env=build_env)
        config = spec['platforms'][target]
        env = dict(os.environ, PATH=f"{venv / 'bin'}:{os.environ['PATH']}",
                   MACOSX_DEPLOYMENT_TARGET=config['deployment_target'],
                   SOURCE_DATE_EPOCH=str(source['epoch']),
                   CMAKE_BUILD_PARALLEL_LEVEL=os.environ.get('CMAKE_BUILD_PARALLEL_LEVEL', '8'),
                   CMAKE_ARGS=f"-DCMAKE_OSX_DEPLOYMENT_TARGET={config['deployment_target']} -DMLX_METAL_VERSION={config['metal_version']}")
        for component in ('FRONTEND', 'BACKEND'):
            env['MLX_BUILD_FRONTEND_PACKAGE'] = str(int(component == 'FRONTEND'))
            env['MLX_BUILD_BACKEND_PACKAGE'] = str(int(component == 'BACKEND'))
            if component == 'FRONTEND':
                run([python, 'setup.py', 'clean', '--all'], cwd=checkout, env=env)
                run([python, '-m', 'build', '--wheel', '--no-isolation', '--outdir', directory.resolve()], cwd=checkout, env=env)
            else:
                # The frontend build compiled both halves. Upstream's wheel filter
                # excludes backend files only from the wheel staging directory.
                run([python, 'setup.py', 'bdist_wheel', '--skip-build', '--dist-dir', directory.resolve()], cwd=checkout, env=env)
        build_info = {'kind': 'upstream-source', 'version_overlay': 'setup.py package version only',
                      'xcode': subprocess.check_output(['xcodebuild', '-version'], text=True).strip(),
                      'build_packages': subprocess.check_output([str(python), '-m', 'pip', 'freeze'], text=True).splitlines()}
    else:
        raise ValueError(f"Unsupported MLX source kind: {source['kind']}")
    receipt = {'schema_version': 1, 'runtime_spec_sha256': sha256(SPEC_PATH),
               'version': spec['version'], 'source': source, 'platform': target,
               'target': spec['platforms'][target], 'wheels': wheel_records(directory, spec, target),
               'build': build_info}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    verify(directory, spec, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['version', 'minimum-os', 'prepare', 'verify'])
    parser.add_argument('--platform')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--build-root', type=Path)
    args = parser.parse_args()
    spec = json.loads(SPEC_PATH.read_text())
    if args.command == 'version':
        print(spec['version'])
        return
    if args.platform not in spec['platforms']:
        parser.error('Select a platform from mlx-runtime.json')
    if args.command == 'minimum-os':
        print(spec['platforms'][args.platform]['minimum_system_version'])
        return
    if args.output is None:
        parser.error('--output is required')
    if args.command == 'prepare':
        if spec['source']['kind'] == 'git' and args.build_root is None:
            parser.error('--build-root is required for a source build')
        import fcntl
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / '.prepare.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            prepare(args.output, spec, args.platform, args.build_root)
    else:
        verify(args.output, spec, args.platform)


if __name__ == '__main__':
    main()
