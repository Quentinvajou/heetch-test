# heetch-test

Welcome to heetch-test !



This repo is aims at solving a business case for [Heetch](https://www.heetch.com/fr). Let's see what it contains :



Table of Contents
-----------------

  * [Context](#context)
  * [Getting started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
    
  * [How it works](#how-it-works)
    * [Get dataset](#get-dataset)
    * [Run analysis](#run-analysis)
    * [Run training](#run-training)
    * [Cleaning](#cleaning)



## Context

In the process of recruitment [Heetch](https://www.heetch.com/fr) asks applicants to provide a solution for business case. This repo gives an approach to both data analysis and modelisation of a solution. You can follow the instructions of installation to run the solution. Analysis of the data can be found in the wiki.



## Getting started

### Prerequisites

You'll only need to have [Docker](https://www.docker.com/) up and running. Everything will be executed within docker containers.

### Installation

Clone the repo

```
git clone https://github.com/Quentinvajou/heetch-test.git
```



## How it works

### Get dataset

First we build the image that responsible for data setup :

```bash
docker build -f Dockerfiles/Dockerfile_setup -t heetch_setup .
```



Then execute the container :

* on windows

```bash
docker run -v %cd%:/heetch-test --name heetch_setup --rm -d heetch_setup
```

* on linux and mac

```bash
docker run --user "$(id -u):$(id -g)" -v $(pwd):/heetch-test --name heetch_setup --rm -d heetch_setup
```



### Run analysis

For the analysis and training we'll only need to build one image :

```bash
docker build -f Dockerfiles/Dockerfile_ds -t heetch_ds .
```



Then execute the container to run the analysis tool :

* on windows

```bash
docker run -v %cd%:/heetch-test -p 8501:8501 --name heetch_ds_analytics --rm -d heetch_ds streamlit run src/analytics/__main__.py
```

* on linux and mac

```bash
docker run -v $(pwd):/heetch-test -p 8501:8501 --name heetch_ds_analytics --rm -d heetch_ds streamlit run src/analytics/__main__.py
```

You can now connect on http://localhost:8501/ to explore the datasets. **The analysis content is available in the wiki**.



### Run Training

To train a model you can run this command :

* on windows

```bash
docker run -v %cd%:/heetch-test -e DATASET_SAMPLING_FRACTION=.1 --name heetch_ds_training --rm -ti heetch_ds python src/modeling/__main__.py
```

* on linux and mac

```bash
docker run -v $(pwd):/heetch-test -e DATASET_SAMPLING_FRACTION=.1 --name heetch_ds_training --rm -ti heetch_ds python src/modeling/__main__.py
```

Feature engineering was not optimised. It will take a few minutes to run the first time. A version of the dataset engineered is kept to allow for faster iteration on training and testing.

#### Environment variables

Using environment variable we can pass parameters to modify the training of the model. 

```bash
-e DATASET_SAMPLING_FRACTION=.1 #  sample the original dataset to allow for faster training. ]0,1]
-e TRAINING_TYPE=naive #   help the reader of the analysis to reproduce the different steps of the modelisation. [naive, ...]
```

To have the most stable model please use DATASET_SAMPLING_FRACTION=1. It may take a few minutes on the first training but the preprocessed data is stored and allows for faster training when modifying TRAINING_TYPE.

### Cleaning

To get rid of the images just use :

```bash
docker rmi -f heetch_setup heetch_ds
```

 

