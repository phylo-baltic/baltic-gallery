# SARS-CoV-2 pandemic

### Source of data

The datasets used here come from multiple sources. The country-annotated MCC tree is derived from [Dudas _et al_. (2021)](https://www.nature.com/articles/s41467-021-26055-8), raw data of lineage counts was taken from [Yu _et al_. (2024)](https://journals.plos.org/plospathogens/article?id=10.1371/journal.ppat.1012090), while the rest are taken from Genbank (accessions included in sequence names).


### Visualisation(s)

The code here shows how the automatic parsing of travel-aware BEAST discrete phylogeography outputs is parsed in baltic as well as SNP alignment visualisations seen in Fig 1 of [Dudas _et al_. (2021)](https://www.nature.com/articles/s41467-021-26055-8), Fig 2 of [Pirnay _et al_. (2021)](https://www.mdpi.com/1999-4915/13/7/1359), Fig 3A of [Holtz _et al_. (2024)](https://link.springer.com/article/10.1186/s12879-024-09967-w). 

The code for estimating lineage frequencies is taken and adapted from [augur's](https://github.com/nextstrain/augur) `frequency_estimator.py` which allows assignment of multiple observations (collection dates) to branches of an abstract SARS-CoV-2 lineage tree. The Muller plot is derived from approximately 1.3 million genomes.