# BioComet


## Community Explorer for Multi-omics daTa

<img align="right" width="450" height="450" src="https://github.com/NiRuff/GithubMedia/blob/main/BioCometLogo_cropped.png?raw=true">

BioComet is an advanced bioinformatics tool designed to facilitate the analysis of protein-protein interactions (PPIs) and the exploration of biological networks. Built as an object-oriented Python module, BioComet leverages the STRING database to provide users with powerful capabilities for community detection, functional annotation, and network visualization.

### Principle
Large gene lists or gene lists emerging from the analysis of complex conditions can functionally point to a variety of underlying biological pathways. 
When analyzing all of these data as one input in a functional enrichment analysis, this can lead to a facilitation of the unspecific top-level terms obscuring more detailed underlying findings.
E.g., this set of 139 genes emerging as commonly appearing in several neurodegenerative diseases 
([Ruffini et al., Cells 2020](https://doi.org/10.3390/cells9122642)) shows a functional enrichment that is most significantly associated with the vague top-level term "Disease".

If assuming that the transcriptomic commonalities between these neurodegenerative diseases is derived from a variety of processes, a further observation of this large gene set as subnetworks might shed more light into the fine underlying pathways.

![BioCometOverview](https://user-images.githubusercontent.com/50486014/238303370-5f6a0280-ef52-4dba-8f1a-7762256f83c6.png)

## Features

- **Community Detection**: Utilize the Louvain and Leiden algorithms for robust community detection in PPI networks.
- **Functional Annotation**: Perform comprehensive functional annotation using databases like KEGG, Reactome, MetaCyc, the EBI Complex Portal, and Gene Ontology Complexes.
- **Network Visualization**: Generate visual representations of PPI networks, including specific communities, with support for various visualization methods.
- **Integrated Hub Node Detection**: Seamlessly identify hub nodes within communities or across the entire PPI graph.

## Installation

biocomet is distributed via PyPi so it can easily be installed with pip. However, in case problems might arise, we also shortly showcase how to install it in a fresh conda environment.

### Using pip

BioComet can be easily installed via pip:

```bash
pip install biocomet
```

If any issues arise, trying creating a new conda environment with Pyton >=3.6 and installing biocomet in that fresh environment.
This could e.g. look like this:

```bash
conde create -n biocomet_env python=3.10
conda activate biocomet_env
pip install biocomet
```

## Usage

After installation, you can start using BioComet by importing it into your Python projects:

```python
import pandas as pd
import biocomet as bc

# either draw data from our example section or use your own data

# loading your data could e.g. look like this
my_DEG_results = pd.read('path/to/my/DEG_results.csv')


# Example usage
ppi_graph = bc.PPIGraph(gene_list=my_DEG_results['GeneID'], reg_list=my_DEG_results['logFC'])
```
For more details see the example section in which we offer three different use-cases to browse through.

## Contributing

We welcome contributions to BioComet! If you have suggestions, bug reports, or contributions, please submit them as issues or pull requests on GitHub.

## Citing BioComet

If you use BioComet in your research, please cite our paper: 

tbd

