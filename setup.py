#!/usr/bin/env python

from setuptools import setup

setup(name='epi',
      version='0.1',
      description='Emergent property inference.',
      author='Sean Bittner',
      author_email='srb2201@columbia.edu',
      install_requires=['tensorflow>=2.4.0',
                        'tensorflow-probability>=0.11.1',
                        'scikit-learn',
                        'numpy',
                        'pandas',
                        'seaborn',
                        'matplotlib',
                        'pytest-cov'],
      packages=['epi'],
     )
