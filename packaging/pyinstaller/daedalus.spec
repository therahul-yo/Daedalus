# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

# PyInstaller resolves relative script/data paths against the SPEC file's
# directory (packaging/pyinstaller/), not the invocation cwd — anchor to the
# repo root explicitly or the release build dies with "script not found".
ROOT = Path(SPECPATH).resolve().parent.parent

block_cipher = None

# Import daedalus as a real package (via pathex), NOT as bundled source data:
# shipping daedalus/ as `datas` made the frozen import of daedalus.cache.store
# fail ("Modified Name"/ImportError) because the submodules weren't collected.
# collect_submodules pulls every daedalus.* module into the archive.
a = Analysis(
    [str(ROOT / 'daedalus' / 'cli.py')],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    hiddenimports=collect_submodules('daedalus') + [
        'mlx',
        'mlx.core',
        'mlx.nn',
        'mlx.optimizers',
        'mlx_lm',
        'mlx_lm.generate',
        'mlx_lm.sample',
        'mlx_lm.utils',
        'mlx_lm.tokenizer_utils',
        'mlx_lm.models',
        'mlx_lm.models.llama',
        'mlx_lm.models.mistral',
        'mlx_lm.models.mixtral',
        'mlx_lm.models.phi',
        'mlx_lm.models.gemma',
        'mlx_lm.models.qwen2',
        'mlx_lm.models.qwen3',
        'mlx_lm.models.phi3',
        'mlx_lm.models.llama3',
        'tokenizers',
        'transformers',
        'transformers.models.llama',
        'transformers.models.mistral',
        'transformers.models.mixtral',
        'transformers.models.phi',
        'transformers.models.gemma',
        'transformers.models.qwen2',
        'transformers.models.qwen3',
        'transformers.models.phi3',
        'transformers.models.llama3',
        'fastapi',
        'uvicorn',
        'uvicorn.protocols.http.h11_impl',
        'uvicorn.protocols.http.httptools_impl',
        'uvicorn.protocols.websockets',
        'uvicorn.lifespan',
        'httpx',
        'pydantic',
        'pydantic_core',
        'pydantic_core._pydantic_core',
        'pydantic.json_schema',
        'pydantic.types',
        'pydantic.functional_validators',
        'pydantic.functional_serializers',
        'pydantic.main',
        'pydantic.fields',
        'pydantic.config',
        'daedalus.observability',
        'daedalus.audit',
        'daedalus.cache.cli',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'test',
        'tests',
        'pytest',
        'unittest',
        'doctest',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='daedalus',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[
        'vcruntime140.dll',
    ],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)