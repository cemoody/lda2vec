from setuptools import setup
from setuptools import find_packages
import os

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

# If building on RTD, don't install anything
if os.environ.get('READTHEDOCS', None) == 'True':
    install_requires = []

kw = dict(
    name='lda2vec',
    version='0.1',
    description='Tools for interpreting natural language',
    author='Christopher E Moody',
    author_email='chrisemoody@gmail.com',
    install_requires=install_requires,
    packages=find_packages(),
    url='')

setup(**kw)
