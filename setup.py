from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

kw = dict(
    name='lda2vec',
    version='0.1',
    description='Tools for interpreting natural language',
    author='Christopher E Moody',
    author_email='chrisemoody@gmail.com',
    install_requires=install_requires,
    url='',
    packages = find_packages(),
)

setup(**kw)
