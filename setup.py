#!/usr/bin/env python

from distutils.core import setup


install_requires = [
    'chainer',
    'numpy',
    'spacy'
    'sklearn']


setup(name='lda2vec',
      version='0.1',
      description='Tools for interpreting natural language',
      author='Christopher E Moody',
      author_email='chrisemoody@gmail.com',
      install_requires=install_requires,
      url='')
