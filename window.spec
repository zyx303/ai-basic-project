# -*- mode: python ; coding: utf-8 -*-
import os
import torch

block_cipher = None

a = Analysis(
    ['window.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('model', 'model'),
        ('utils', 'utils'),
    ],
    hiddenimports=[
        'model.SentimentClassifier',
        'utils.data_loader',
        'utils.train_utils',
        'transformers',
        'sklearn.metrics',
        'sklearn.preprocessing',
        'torch',
        'torchvision',
        'matplotlib',
        'pandas',
        'numpy',
        'PyQt5.QtCore',
        'PyQt5.QtWidgets',
        'PyQt5.QtGui',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# 动态获取PyTorch库路径
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
if os.path.exists(torch_lib_path):
    pyd = Tree(torch_lib_path, prefix='torch\\lib\\', excludes=['*.pdb'])
    a.datas += pyd

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SentimentAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # 设置为False以隐藏控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None  # 可以添加图标文件路径
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SentimentAnalyzer'
)
