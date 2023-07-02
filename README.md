# DRBpred
DRBpred: A Sequence-based Machine Learning Method to effectively predict DNA and RNA Binding Residues

# Table of Content

- [Getting Started](#getting-started)
- [Datasets](#datasets)
- [Prerequisites](#prerequisites)
- [Download and install code](#download-and-install-code)
- [Authors](#authors)
- [References](#references)

# Getting Started
 

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Datasets
The dataset can be found in the Dataset/FullDataset directory. The dataset is collected from [1].



### Prerequisites

We have tested DRBpred on Ubuntu 20.04. You would need to install the following software before replicating this framework in your local or server machine. 

1. pyenv latest version
    ```
    curl https://pyenv.run | bash
    exec $SHELL
    ```
    For more details, visit: https://github.com/pyenv/pyenv-installer

2. Python version 3.9.5

    ```
    pyenv install miniconda3-3.9-4.10.3
    pyenv local miniconda3-3.9-4.10.3 
    ```

    For more details, visit: https://github.com/pyenv/pyenv

    Alternatively, Python version 3.9.5 can be installed using Anaconda.
    
    ```
    conda create -n py39 python=3.9.5 
    conda activate py39
    ``` 

3. Poetry version 1.3.2

    ```
    curl -sSL https://install.python-poetry.org | python3 - --version 1.3.2
    ```
    For more details, visit: https://python-poetry.org/docs/

4. Docker

    The feature extraction part depends on many other tools. So we created a docker image to extract features easily without any setup. It might take quite some time to get all the features.

6. Protein Databases

    The tool depends on the nr and uniclust30_2017_04 databases. The database should be placed in the following directory structure. Alternatively, it is possible to pass the database path as parameters.

```
project
│─── README.md    
│
└─── script
    └─── Databases
           └─── nr
            └───uniclust30_2017_04

```
## Download and install code

- Retrieve the code

```
git clone https://github.com/wasicse/DRBpred.git

```

To run the program, first install all required libraries by running the following command:

```

POETRY_VIRTUALENVS_IN_PROJECT="true"
poetry install

```

Then execute the following command to run DRBpred from the script directory on the example dataset. You need to change the input of the Dataset/example directory to get prediction for new protein sequences and replace DATABASE_PATH with the absoutue path of the databases e.g, "/home/wasi/DRBpred/script/Databases/"

```
cd script
poetry run python run_DRBpred.py -f "featuresv2" -o "./output/" -d "DATABASE_PATH"
```

- Finally, check **output** folder for results. The output directory contains predicted lebels with probabilities for each residues.


## Authors

Md Wasi Ul Kabir, Duaa Mohammad Alawad,  Md Tamjidul Hoque. For any issue please contact: Md Tamjidul Hoque, thoque@uno.edu 

## References

1. Yan, J. and Kurgan, L.J.N.a.r. DRNApred, fast sequence-based method that accurately predicts and discriminates DNA-and RNA-binding residues. 2017;45(10):e84-e84.

