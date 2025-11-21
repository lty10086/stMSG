# stMSG, a unified framework that integrates Maximum mean discrepancy (MMD) mapping, structure-aware graph augmentation, and embedding-guided graph autoencoder (GAE) integration.

All processed datasets can be downloaded at Synapse (https://www.synapse.org/Synapse:syn71726608/files/ and https://www.synapse.org/Synapse:syn64421787/files/).

## Datasets

All datasets used in this study are publicly available.

- Data sources and detailed information are provided in [Supplementary_Table_1](Supplementary_Table_1). After downloading the data, please refer to the processing steps outlined in [Data Processing README.txt](Data_Processing_README.txt) and execute the code in [Data Processing.py](Data_Processing.py) to perform the analysis and obtain clustering results.
- All processed datasets can be downloaded at [Zenodo](https://zenodo.org) and [Synapse](https://www.synapse.org).

The datasets should be organized in the following structure:
|-- dataset
    |-- STdata.h5ad
    |-- scRNAdata.h5ad
