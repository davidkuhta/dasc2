# Copyright 2017 David Kuhta & Anshul Sacheti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module setuptools script."""

from __future__ import absolute_import
from __future__ import print_function

from setuptools import setup

description = """dASC2 - StarCraft II Data Analytics

dASC2 is a Data Analytics toolset for StarCraft II written in Python.

Read the README at https://github.com/davidkuhta/dasc2 for more information.
"""

extras = {
    'with-progress-bar': ['tqdm>=4.19.4']
}

setup(
    name='daSC2',
    version='0.2.2-dev3',
    description='Data Analytics Library for StarCraft II',
    long_description=description,
    author='David Kuhta',
    author_email='davidkuhta@gmail.com',
    license='Apache License, Version 2.0',
    keywords=['StarCraft', 'StarCraft II', 'StarCraft 2', 'StarCraft AI', 'data analytics', 'SC2Replay'],
    url='https://github.com/davidkuhta/dasc2',
    packages=[
        'dasc2',
        'dasc2.bin',
        'dasc2.lib',
        'dasc2.agent'
    ],
    install_requires=[
        'future',
        'mpyq',
        'pysc2'
    ],
    entry_points={
        'console_scripts': [
            'dasc2_download = dasc2.bin.download_replays:main',
            'dasc2_extract = dasc2.bin.extract_replays:main',
            'dasc2_filter = dasc2.bin.filter_replays:main',
            'dasc2_states = dasc2.bin.generate_states:main',
            'dasc2_build_orders = dasc2.bin.build_orders:main',
            'dasc2_template_agent = dasc2.bin.template_agent:main',
        ],
    },
    extras_require=extras,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    include_package_data = True
)
