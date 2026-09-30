# -*- mode: python ; coding: utf-8 -*-
import os
import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata, get_package_paths

# Transformers' custom model loader inspects Python source at runtime.
datas = collect_data_files('transformers', include_py_files=True)
datas += collect_data_files('timm', include_py_files=True)
datas += collect_data_files('image_dataset_studio', includes=['assets/*.png'])
for package in ['transformers', 'timm', 'huggingface_hub', 'safetensors', 'torch', 'torchvision']:
    datas += copy_metadata(package)
hiddenimports = collect_submodules('timm')
for model_package in ['florence2', 'qwen3_vl', 'llava']:
    hiddenimports += collect_submodules(f'transformers.models.{model_package}')
hiddenimports += ['transformers.models.auto', 'transformers.image_processing_base',
                  'transformers.image_utils', 'transformers.pipelines',
                  'torchvision.transforms.functional', 'torch.utils.checkpoint']
# torchvision 0.29 uses *_stable.pyd; older PyInstaller hooks only collect _C.pyd.
_, torchvision_dir = get_package_paths('torchvision')
binaries = [(str(p), 'torchvision') for p in Path(torchvision_dir).iterdir()
            if p.suffix.lower() in {'.pyd', '.dll'}]
a = Analysis(['run_app.py'], pathex=['src'], binaries=binaries, datas=datas,
             hiddenimports=hiddenimports, hookspath=[], hooksconfig={},
             runtime_hooks=[], excludes=['tkinter', 'matplotlib', 'IPython', 'pytest'])
# Qt's Windows wheel uses the OS ICU interface. An unrelated ICU on PATH
# (e.g. Poppler) exports version-suffixed symbols and must not shadow it.
a.binaries = [entry for entry in a.binaries
              if not re.fullmatch(r'icu(?:uc|in|dt\d+)\.dll', Path(entry[0]).name, re.IGNORECASE)]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='ImageDatasetStudio',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=os.environ.get('IDS_BUILD_CONSOLE') == '1',
          icon=str(Path(SPECPATH) / 'src/image_dataset_studio/assets/app_icon.ico'))
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='ImageDatasetStudio')
