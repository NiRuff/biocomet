from setuptools import setup, find_packages

setup(
    name='biocomet',
    version='0.1.7',
    author='Nicolas Ruffini',
    author_email='nicolas.ruffini@lir-mainz.de',
    description='A brief description of the biocomet package',
    url='https://github.com/NiRuff/COMET/tree/master_kegg',
    packages=find_packages(),
    package_data={'biocomet': ['data/*.json', 'data/*.tsv', 'data/*.gz']},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.19.2',
        'pandas>=1.2.0',
        'scikit-learn>=0.24',
        'matplotlib>=3.4',
        'seaborn>=0.11',
        'beautifulsoup4>=4.9',
        'requests>=2.25',
        'jupyter>=1.0.0',
        'ipython>=7.22.0',
        'python-igraph>=0.10.9',
        'leidenalg>=0.10.2',
        'nxviz>=0.7.4',
        'tqdm>=4.66.2',
        'wordcloud>=1.9.3',
        'networkx>=3.2.1',
        'python-louvain>=0.16',
    ]
)
