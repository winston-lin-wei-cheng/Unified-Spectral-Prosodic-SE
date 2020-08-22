# Unified Spectral and Prosodic SE
This is a demo implementation of the Unified Spectral and Prosodic Speech Enhancement (SE) framework in the [paper](http://www.apsipa.org/proceedings/2019/pdfs/188.pdf) based on a given small test sample set (i.e., the *'test_samples'* folder). 

# Suggested Environment and Requirements
1. Python 3.6+
2. Ubuntu 18.04
3. Praat (available to download on the [official website](https://www.fon.hum.uva.nl/praat/download_linux.html))
4. keras version 2.2.4
5. tensorflow version 1.14.0
6. librosa version 0.7.0
7. pystoi version 0.2.2

# Prosodic Feature Extraction (Praat) 
Under the *'prosodic_feature_extraction'* folder, the praat script **feat.praat** is utilized to extract pitch and intensity prosodic features via Praat. The python script **feat_comb_praat.py** concatenates these features as a matrix and then output with .mat file.  
1. create *'feat_praat'* folder
2. create *'feat_mat'* folder
3. change directory$ & outdir$ paths in **feat.praat** and run in the terminal with
```
praat feat.praat
```
4. change path in **feat_comb_praat.py** and run in the terminal with
```
python feat_comb_praat.py
```

# How to run
After extracted the prosodic features of desired corpus, we use the **training.py** to train the SE models and evluate PESQ & STOI perofrmance results by the **testing.py**.
1. Parameters in the **training.py** are,
   * -epoch: number of training epochs
   * -model_type: 'JointConcat', 'FixedConcat' and 'MultiTask' are supported
   * -path_dataset: directory of the dataset
   * Notice: if you use the 'FixedConcat' model, you will need to pretrain a FE model by the **pretrain_fe.py**
2. Parameters in the **testing.py** are,
   * -model_type: 'JointConcat', 'FixedConcat' and 'MultiTask' are supported
   * -path_dataset: directory of the dataset
   * -path_output: output directory of the enhanced WAV files

# Reference
If you use this code, please cite the following paper:

Wei-Cheng Lin, Yu Tsao, Fei Chen and Hsin-Min Wang, "Investigation of Neural Network Approaches for Unified Spectral and Prosodic Feature Enhancement" in 2019 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC). IEEE, 2019, pp. 1179–1184.

```
@inproceedings{lin2019investigation,
  title={Investigation of Neural Network Approaches for Unified Spectral and Prosodic Feature Enhancement},
  author={Lin, Wei-Cheng and Tsao, Yu and Chen, Fei and Wang, Hsin-Min},
  booktitle={2019 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)},
  pages={1179--1184},
  year={2019},
  organization={IEEE}
}
```
