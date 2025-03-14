# NNERO

[![Build Status](https://github.com/gaetanfacchinetti/NNERO/actions/workflows/python-package.yml/badge.svg?branch=torch)](https://github.com/gaetanfacchinetti/NNERO/actions?query=branch%3Atorch)


[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-green.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![Static Badge](https://img.shields.io/badge/physics-cosmology-darkblue)](https://en.wikipedia.org/wiki/Cosmology)
[![Static Badge](https://img.shields.io/badge/physics-21cm-yellow)](https://en.wikipedia.org/wiki/Hydrogen_line)

This is **NNERO** (**N**eural **N**etwork **E**mulator for **R**eionization and **O**ptical depth), a fast adaptative tool to emulate reionization history using a simple neural network architecture. 

The current default networks implemented have been trained on data generated with **[21cmCLAST](https://github.com/gaetanfacchinetti/21cmCLAST)**. 

---
> This package is part of a set of codes which can be combined together to produce forecast or constraints from late-time Universe observables (such as 21cm) on exotic scearios of dark matter and more. Some of these packages are forks of previously existing repositories, some have been written from scratch
- [21cmCLAST](https://github.com/gaetanfacchinetti/21cmCLAST) forked from [21cmFAST](https://github.com/21cmfast/21cmFAST)
- [HYREC-2](https://github.com/gaetanfacchinetti/HYREC-2) forked from [this repository](https://github.com/nanoomlee/HYREC-2)
- [MontePython](https://github.com/gaetanfacchinetti/montepython_public) forked from [this repository](https://github.com/brinckmann/montepython_public)
- [21cmCAST](https://github.com/gaetanfacchinetti/21cmCAST)



## How to install NNERO?

NNERO can be installed using pip with the following command
```bash
pip install nnero
```
For a manual installation or development you can clone this repository and install it with
```bash
git clone https://github.com/gaetanfacchinetti/NNERO.git 
pip install -e .
```

## How to use NNERO?

- A detailed documentation is under construction [here](https://gaetanfacchinetti.github.io/docs/NNERO/html/index.html).
- A short tutorial can either be found in the documentation or on the [wiki page](https://github.com/gaetanfacchinetti/NNERO/wiki).

## Contributions

Any comment or contribution to this project is welcome.

## Credits

If you use **NNERO** or the default classifiers / regressor trained using **21cmCLAST** please cite at least one of the following paper that is relevant to your usage:

- G. Facchinetti, *Teaching reionization history to machines:  \\
constraining new physics with early- and late-time probes* (in prep.)
- V. Dandoy, C. Doering, G. Facchinetti, L. Lopez-Honorez, J. R. Schwagereit (in prep.)
- G. Facchinetti, A. Korochkin, L. Lopez-Honorez (in prep.)
