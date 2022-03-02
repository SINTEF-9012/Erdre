[Documentation - Home](../index.md)

## 1. Installation

Developed using Python3.8. You can install the required modules by creating a
virtual environment and install the `requirements.txt`-file (run these commands
from the main folder):

```
mkdir venv
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

To get a plot of the neural network architecture, the following software needs
to be installed: [Graphviz](https://graphviz.org/about/).

## Alternative way of installing requirements (not recommended)

As an alternative you can install the required modules by running the command below, but be aware that this may cause problems due to mismatching version requirements.

```
pip3 install dvc pandas pandas-profiling sklearn xgboost tensorflow tensorflow-probability edward2 plotly nonconformist
```


Next: [Quickstart](02_quickstart.md)
