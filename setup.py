from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name='extend',
    version='0.1.0',
    description='Code for Extend, don\'t rebuild: Phrasing conditional graph modification as autoregressive sequence labeling (EMNLP\'21)',
    url='https://github.com/leonweber/extend',
    author='Leon Weber',
    author_email='leonweber@posteo.de',
    license='MIT',
    packages=find_packages(),
    install_requires=required,
)