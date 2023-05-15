# COMET
COMET: Community Explorer for Multi-omics daTa


Jupyter notebook script for analyzing sets of genes/proteins underlying complex traits. The subdivion into PPI-derived closely related subnetworks (Communities) using the louvain algorithm enablers the identification of finer underlying pathways.

E.g., this set of 139 genes emerging as commonly appearing in several neurodegenerative diseases (Ruffini, N.; Klingenberg, S.; Schweiger, S.; Gerber, S. Common Factors in Neurodegeneration: A Meta-Study Revealing Shared Patterns on a Multi-Omics Scale. Cells 2020, 9, 2642. https://doi.org/10.3390/cells9122642) shows a functional enrichment that is most significantly associated with the vague term "Disease".

If assuming that the transcriptomic commonalities between these neurodegenerative diseases is derived from a variety of processes, a further observation of this large gene set as subnetworks might shed more light into the fine underlying pathways.

![COMEToverview](https://user-images.githubusercontent.com/50486014/238303370-5f6a0280-ef52-4dba-8f1a-7762256f83c6.png)

All packages necessary can be installed using pip.

Functional Enrichment Databases to consider can be set in the script.
