<h1 align="center">Self-Supervised Learning with Probabilistic Density Labeling for Rainfall Probability Estimation</h1>

<div align='center'>
    <strong>Jun Ha Lee</strong></a><sup> 1,3</sup>&emsp;
    <a href='https://scholar.google.com/citations?hl=ko&user=gKrLgVUAAAAJ' target='_blank'><strong>So Jung An</strong></a><sup> 2</sup>&emsp;
    <strong>Su Jeong You</strong><sup> 3</sup>&emsp;
    <a href='https://scholar.google.com/citations?hl=ko&user=Ntx5VRIAAAAJ' target='_blank'><strong>Nam Ik Cho</strong></a><sup> 1</sup>&emsp;
</div>

<div align='center'>
    <sup>1 </sup>Seoul National University&emsp; <sup>2 </sup>Korea Institute of Atmospheric Prediction Systems&emsp; <sup>3 </sup>Korea Institute of Industrial Technology&emsp;
</div>

## 🔔 Updates
- `2024/10/29`: SSLPDL has been accepted at WACV 2025! 🎊

<br>
<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Numerical weather prediction (NWP) models are fundamental in meteorology for simulating and forecasting the behavior of various atmospheric variables. The accuracy of precipitation forecasts and the acquisition of sufficient lead time are crucial for preventing hazardous weather events. However, the performance of NWP models is limited by the nonlinear and unpredictable patterns of extreme weather phenomena driven by temporal dynamics. In this regard, we propose a Self-Supervised Learning with Probabilistic Density Labeling (SSLPDL) for estimating rainfall probability by post-processing NWP forecasts. Our post-processing method uses self-supervised learning (SSL) with masked modeling for reconstructing atmospheric physics variables, enabling the model to learn the dependency between variables. The pre-trained encoder is then utilized in transfer learning to a precipitation segmentation task. Furthermore, we introduce a straightforward labeling approach based on probability density to address the class imbalance in extreme weather phenomena like heavy rain events. Experimental results show that SSLPDL surpasses other precipitation forecasting models in regional precipitation post-processing and demonstrates competitive performance in extending forecast lead times.
</details>

## Intro
This repo, named **SSLPDL**, contains the official PyTorch implementation of our paper Self-Supervised Learning with Probabilistic Density Labeling for Rainfall Probability Estimation.

## 🔥 Getting Started
### 1. Clone the code and prepare the environment 🔧
```bash
git clone https://github.com/junha425/SSLPDL
cd SSLPDL

# create env using conda
conda create -n SSLPDL python==3.10.12
conda activate SSLPDL
# install dependencies with pip
pip install -r requirements.txt
```
### 2. Download pre-trained weights
```bash
# first, ensure git-lfs is installed, see: https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage
git lfs install
# clone the weights
Coming soon..
```

### 3. Run the SSLPDL 🚀
Before running, ensure you configure the necessary settings through the `run.py` file.

#### Pre-training
```bash
# If you want to train the pre-trained model, just run the [main.py]. 
python main.py
```
#### Fine-tuning
```bash
# After training the pre-trained model, just run the [main.py] with the [checkpoint] (file).
# The [checkpoint] (file) under the [pre_path] (directory) is required!
python main.py 
```

## TODO
- [ ] Preprint Paper
- [X] Code
- [ ] Checkpoint

## Contact
joonha4670@gmail.com

## Acknowledgements
We would like to thank the contributors of the [InternImage](https://github.com/OpenGVLab/InternImage) repository for their open research and contributions.

## Citation
If you find SSLPDL useful for your research, welcome to this repo and cite our work using the following BibTeX:
