import setuptools


setuptools.setup(
    name="vqda",
    version="1.0",
    author="sangcamap",
    author_email="",
    description="Vietnamese question DA",
    long_description="Vietnamese question augmentation",
    url="https://github.com/sangcamap/vqda.git",
    packages=setuptools.find_packages(),
    install_requires=['transformers', 'sentencepiece', 'underthesea', 'gensim', 'random', 'simplet5'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
)