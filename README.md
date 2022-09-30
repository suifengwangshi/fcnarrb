# FCNARRB
Locating Transcription factor binding sites (TFBSs) by fully convolutional network
## Requirements

+ Pytorch 1.0.1
+ Python 3.6.8
+ CUDA 9.0
+ Python packages: biopython, sklearn

## Data preparation
(1) Download hg19.fa from http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/, and put it into /your path/hg19/.

(2) Download conservation information from https://hgdownload.soe.ucsc.edu/goldenpath/hg19/phyloP100way/hg19.100way.phyloP100way/, and put it into /your path/conservation/.

(3) Pre-processing datasets.
+ Usage:
  ```
  4 channels: bash process.sh chip_processing.py <data path>
  5 channels: bash process.sh chip_processing_1.py <data path>
  ```

## Implementation 
**Running FCN/FCNA/FCNARRB**
+ Usage: 
  ```
  bash run.sh <data path> <model path> <data channels>
  ```
 
**Locating TFBSs**
+ Usage: 
  ```
  bash locate.sh <data path> <trained model path>
  ```
**Predicting motifs**
+ Usage: 
  ```
  bash motif.sh <data path> <trained model path>
  ```  
