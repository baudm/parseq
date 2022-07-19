#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='strhub',
    version='1.0.0',
    description='Scene Text Recognition Model Hub: A collection of deep learning models for Scene Text Recognition',
    author='Darwin Bautista',
    author_email='baudm@users.noreply.github.com',
    url='https://github.com/baudm/parseq',
    install_requires=['torch~=1.10.2', 'pytorch-lightning~=1.4.9', 'timm~=0.4.12'],
    packages=find_packages(),
)
