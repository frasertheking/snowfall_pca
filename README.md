<div align="center">

![logo](https://github.com/frasertheking/snowfall_pca/blob/main/images/title.png?raw=true)

Primary Modes of Northern Hemisphere Snowfall Particle Size Distributions, maintained by [Fraser King](https://frasertheking.com/)

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

</div>

---

## Primary Modes of Northern Hemisphere Snowfall Particle Size Distributions

This project is currently being written into a journal article for the [Journal of the Atmospheric Sciences](https://www.ametsoc.org/index.cfm/ams/publications/journals/journal-of-the-atmospheric-sciences/).

Snowfall is a critical contributor to the global water-energy budget, with important connections to water resource management, flood mitigation, and ecosystem sustainability. This research enhances our understanding of varying snow particle size distributions in the Northern Hemisphere, offering valuable new insights for improving future remote sensing-based snowfall retrieval algorithms. Using a statistical technique called Principal Component Analysis, we found that 95\% of the variability in observed snowfall could be explained by three primary features: how intense the snowfall is, how dense the particles are, and the depth of the storm. We identified six unique snowfall groups, each with its own set of traits, such as the snow being light and fluffy, or heavy and packed. By linking these traits to external environmental observations, we can better understand the driving physical mechanisms within each group.

This code repository holds the microphysical data extraction and processing scripts, the PCA code, plotting/visualization calls and ancillary analysis notebooks. 
![PCA](https://github.com/frasertheking/snowfall_pca/blob/main/images/pca.png?raw=true)

## Data Sources

A Comprehensive Northern Hemisphere Particle Microphysics Dataset from the Precipitation Imaging Package

The data for this project is hosted online on UM's DeepBlue repository: https://deepblue.lib.umich.edu/data/concern/data_sets/kk91fm40r?locale=en

We have collected PIP microphysical data from a variety of measurement locations across the northern hemisphere. Data originally in a proprietary ASCII format has been converted to the more universally recognized NetCDF-4 format for ease of sharing and compatibility within the academic community. The conversion process, undertaken using a combination of bash and Python, ensures broader compatibility with various data analysis tools and platforms. A quality assurance (QA) procedure has been undertaken to ensure the integrity of the data. Post QA, the data is transformed into daily NetCDF-4 files following the Climate and Forecast (CF) conventions (version 1.10) and compressed with a level 2 deflation for optimized file size. Additional details into the data curation process can be found in our journal article publication.

![data overview](https://github.com/frasertheking/snowfall_pca/blob/main/images/fig01.png?raw=true)

## Installation

To perform your own training tests, you'll want to clone this repo, and create a Conda environment with scikit-learn installed.

## Examples

To interact with the processing scripts, please see the commented Python Jupyter notebooks in the /notebooks path of this repo.

A subset example dataset can also be downloaded here: [LINK]

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Note that, as a living project, code is not as clean as it could (should) be, and unit tests need to be produced in future iterations to maintain stability.

## Authors & Contact

- Fraser King, University of Michigan, kingfr@umich.edu
- Claire Pettersen, University of Michigan
- Brenda Dolan, Colorado State University
- Julia Shates, NASA Jet Propulsion Laboratory
- Derek Posselt, NASA Jet Propulsion Laboratory

## Funding
This project was funded by NASA New (Early Career) Investigator Program (NIP) grant at the [University of Michigan](https://umich.edu).

