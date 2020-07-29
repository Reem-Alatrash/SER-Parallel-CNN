# SER-Parallel-CNN

## About
A speech emotion recognition (SER) system for a deep learning master-course. It uses a convolutional neural network (CNN) with parallel convolution layers to classify the emotion embedded within speech signals (both scripted and improvised) into 1 of 4 emotions: happy, sad, angry, neutral. 

Note: The scripts are written in python 2.7

##### Input
The system takes as input pre-processed features extracted from the speech subset of the Interactive Emotional Dyadic Motion Capture (IEMOCAP) database.

##### Output
The system produces 2 files:
1. The model (parallel CNN) is saved as "parallel_cnn_BN.h5"
2. The predictions for both validation/dev and test sets in tab-seperated files. Each prediction file contains the IDs audio signals and their corresponding predicted class.

Example:
MSP-IMPROV-S08A-F05-S-FM02	happy


## Usage
The script is available in two formats:
1. A native Python script (parallel-CNN.py).
2. A Jupyter Notebook file (parallel-CNN.ipynb).

##### Data
Download preprocessed logMel features from the link below and extract the files into the empty data directory/folder.
Link to data: https://www.mediafire.com/file/kq1gsmaw85t02x6/data.zip/file

##### Jupiter Notebook
To use the SER sytem: open the file parallel-CNN.ipynb and run in jupyter notebook.
NOTE: This notebook uses a python 2.7 kernel 

##### Python file
To use the SER sytem: run the python file via line commands or using any python IDE.

Example of running the script via line commands:

```bash
python parallel-CNN.py
```

##### Visualizing Validation
Run the notbook Parallel-CNN-self-validate.ipynb to see the validation results and visualize the accuracy and loss.