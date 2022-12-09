# tabulardl
Tabular neural network model exploration

```bash
conda env remove -n tabulardl
conda create -n tabulardl python
conda activate tabulardl
pip3 install ipython
pip3 install numpy
pip3 install scikit-learn
conda install pytorch -c pytorch

pip3 install pytest
pip3 install coverage
```

```bash
coverage run -m pytest && coverage report -m

```

* SASRec gitub. Contains links and data for teh . https://github.com/kang205/SASRec
* CARCA: https://arxiv.org/abs/2204.06519v1
