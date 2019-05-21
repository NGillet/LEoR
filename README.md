# Learning the Epoch of Reionization - LEoR

Code for the paper : https://arxiv.org/pdf/1805.02699.pdf
A Convolutional Neural Network (CNN) taking 2D slice of light-cones (LC) and recover the 21cmFast input parameters.

## Getting Started

The purpose of this repo is mainly a personal saving of the code. 

### Prerequisites

It need the database : LC_SLICE10_px100_2200_N10000_randICs_train.hdf5 (~70GB). Ask me for the data.
The code has been run on SNS-Nefertem

### Installing

- jupyter notebook
- keras
- theano (the paper as been made with theano) (or tensorflow: the code as been optimize after the publication to GPU for futur study)

## Running

- set the parameters of the CNN : 
    - param_all4_2D_smallFilter_1batchNorm_multiSlice.py
    - the name of the parameter file can be change in CNN_lightcone_2D.py
- Learning:
```
python CNN_lightcone_2D.py
```
- Analisis and plots:
  - jupyter notebook : CNN_analyseNet.ipynb
  - will load the learned CNN and validation/testing dataset for the plots. 
- The already trained CNN is available in CNN_analyseNet.ipynb

## Authors

* **Nicolas Gillet** - [LEoR](https://github.com/NGillet/LEoR)

## License

## Acknowledgments

* The database has been made by **Bradley Greig**
