# -*- coding: utf-8 -*-
# @Time    : 2024/8/27 22:47
# @Project : ChatTTSPlus
# @FileName: setup.py

import os
from setuptools import setup, find_packages

version = "v1.0.0"

setup(
    name="chattts_plus",
    version=os.environ.get("CHATTTS_PLUS_VER", version).lstrip("v"),
    description="",
    long_description=open("README_EN.md", encoding="utf8").read(),
    long_description_content_type="text/markdown",
    author="wenshao",
    author_email="wenshaoguo1026@gmail.com",
    url="https://github.com/warmshao/ChatTTSPlus",
    packages=[
        'chattts_plus',
        'chattts_plus.models',
        'chattts_plus.pipelines',
        'chattts_plus.commons',
    ],
    license="AGPLv3+",
    install_requires=[
        "numba",
        "numpy<2.0.0",
        "pybase16384",
        "torch>=2.1.0",
        "torchaudio",
        "tqdm",
        "transformers>=4.41.1",
        "vector_quantize_pytorch",
        "vocos"
    ]
)
