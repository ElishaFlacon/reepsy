#!/usr/bin/env python

from io import open
from setuptools import setup

"""
:authors: ElishaFlacon
:copyright: (c) 2023 ElishaFlacon
"""

version = '0.0.1'

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='reepsy',
    version=version,
    author='ElishaFlacon',
    author_email='',
    description=(
        u'Библиотека для '
        u'Club House (clubhouse.io API wrapper)'
    ),
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/ElishaFlacon/reepsy',
    download_url='https://github.com/ElishaFlacon/reepsy/archive/main.zip',

    license='Apache License, Version 2.0, see LICENSE file',

    packages=['club_house_api'],
    install_requires=['aiohttp', 'aiofiles'],

    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Programming Language :: Python :: Implementation :: CPython',
    ]
)
