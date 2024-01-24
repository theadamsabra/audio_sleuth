from setuptools import setup, find_packages
import codecs 

VERSION = '0.0.2'

with codecs.open("README.md", encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

setup(
    name='audio_sleuth',
    version=VERSION,
    author='theadamsabra (Adam Sabra)',
    description='an open-source framework for detecting audio generated from generative systems',
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['torch', 'torchaudio', 'librosa', 'numpy']
)