"""Shared local workspace paths for handover tools.

CLI paths override inherited paths. An explicit --root resets component defaults.
Child tools inherit resolved paths through namespaced environment variables.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

NAMES = {'root':'ZERO_RISK_ROOT', 'skill':'ZERO_RISK_SKILL',
         'tools':'ZERO_RISK_TOOLS', 'deliverables':'ZERO_RISK_DELIVERABLES',
         'evidence':'ZERO_RISK_EVIDENCE'}
DEFAULTS = {'skill':'02_活副本_Skill', 'tools':'03_工具链',
            'deliverables':'01_交付物', 'evidence':'04_远端证据'}


def resolve_paths(overrides=None):
    overrides = overrides or {}
    root = Path(overrides.get('root') or os.environ.get(NAMES['root'])
                or Path(__file__).resolve().parent.parent).expanduser().resolve()
    result = {'root':str(root)}
    for name, default in DEFAULTS.items():
        value = overrides.get(name)
        if not value and not overrides.get('root'):
            value = os.environ.get(NAMES[name])
        path = Path(value or default).expanduser()
        result[name] = str((root / path).resolve() if not path.is_absolute() else path.resolve())
    return SimpleNamespace(**result)


def bootstrap_paths(parse_cli=False):
    overrides = {}
    show = False
    if parse_cli:
        # The CLI and its subprocesses use one encoding, including redirected logs.
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, 'reconfigure'):
                stream.reconfigure(encoding='utf-8', errors='replace')
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        for name in NAMES:
            parser.add_argument('--'+name)
        parser.add_argument('--show-paths', action='store_true')
        args, remaining = parser.parse_known_args()
        overrides = {name:getattr(args,name) for name in NAMES if getattr(args,name)}
        show = args.show_paths
        sys.argv[1:] = remaining
        if '--help' in remaining or '-h' in remaining:
            print('通用路径参数: --root --skill --tools --deliverables --evidence --show-paths')
    paths = resolve_paths(overrides)
    if parse_cli:
        for name, variable in NAMES.items():
            os.environ[variable] = getattr(paths,name)
        os.environ['PYTHONIOENCODING'] = 'utf-8'
    if show:
        print(json.dumps(vars(paths), ensure_ascii=False, indent=2))
        sys.exit(0)
    return paths
