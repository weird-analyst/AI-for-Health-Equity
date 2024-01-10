# AI for Health Equity

This project is my attempt at trying to recreate the results of and extend the research conducted in the following paper:
https://www.nature.com/articles/s41467-020-18918-3


## TO RUN THE CODE:
Make sure to download the data files and place them at their correct position, the link for the download location is given in the Dataset folder.

```
create environment: conda create --name multiethnic python=3.11
conda activate multiethnic
conda install -c pytorch pytorch 
conda install --file requirements.txt
pip install jupyterlab scipy pandas scikit-learn openpyxl numpy matplotlib pytorch-tabnet
```

To run script: python main.py CANCER-TYPE DATA-TYPE SURVIVAL-OUTCOME YEAR TARGET-GROUP