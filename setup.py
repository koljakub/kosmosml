#!/usr/bin/env python3

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    REQUIREMENTS = [line for line in f.read().splitlines() if not line.startswith('--') and not line.startswith('#')]

setup(
    name='kosmosml',
    version='0.1.0',
    author='Jakub Kol',
    author_email='kol.jakub@protonmail.com',
    description='An easy-to-use library for Content-based image retrieval and Image captioning.',
    url='https://github.com/koljakub/kosmos-ml-lib',
    packages=find_packages(),
    install_requires=REQUIREMENTS,
)
