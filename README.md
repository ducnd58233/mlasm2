# **Setup**
- Using anaconda:
```
conda create -n mlasm2 python=3.9
conda activate mlasm2
conda install --file requirements.txt
```
- task1.ipynb: the file of task 1  
- task2.ipynb: the file of task 2

# **Folder Structure**
```
|- model_saved (saving model at best status)
|- models (define the model architecture)
    |- VGG.py
    |- ResNet.py
|- patch_images (dataset)
|- utils
    |- MyDataset.py (using to read image and convert into Tensor)
    |- trainer.py (define training, validation, and testing method)
    |- visualizer.py (define visualization training and validation and testing accuracy / loss)
|- data_labels_extraData.csv (csv dataset)
|- data_labels_mainData.csv (csv dataset)
|- README.md
|- requirements.txt (file contains all of environment using in this project)
|- task1.ipynb (task 1 of machine learning assignment 2)
|- task2.ipynb (task 2 of machine learning assignment 2)
```
