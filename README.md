# heetch-test

Welcome to heetch-test !



This repo is aims at solving a business case for [Heetch](https://www.heetch.com/fr). Let's see what it contains :



[[TOC]]



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

```bash
docker run -v $(pwd):/heetch-test --name heetch_setup --rm -d heetch_setup
```

In Linux use *$(pwd)* to get the path to the repo. Alternatively in :

* Windows Command Line (cmd) use : *%cd%*
* Windows PowerShell use : *${PWD}*



### Run analysis

For the analysis and training we'll only need to build one image :

```bash
docker build -f Dockerfiles/Dockerfile_ds -t heetch_ds .
```



Then execute the container to run the analysis tool :

```bash
docker run -v $(pwd):/heetch-test --name heetch_ds_analytics --rm -d heetch_ds streamlit run src/analytics/__main__.py
```

You can now connect on http://localhost:8501/ to explore the datasets. **The analysis content is available in the wiki**.



### Run Training

To train a model you can run this command :

```bash
docker run -v $(pwd):/heetch-test --name heetch_ds_training --rm -ti heetch_ds python src/modeling/__main__.py
```

Feature engineering was not optimised. It will take a few minutes to run the first time. A version of the dataset engineered is kept to allow for faster iteration on training and testing.



### Cleaning



