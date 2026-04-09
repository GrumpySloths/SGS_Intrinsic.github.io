# SGS-Intrinsic: Semantic-Invariant Gaussian Splatting for Sparse-View Indoor Inverse Rendering (CVPR 2026)

Jiahao Niu, Rongjia Zheng, Wenju Xu, Wei-shi Zheng, and Qing Zhang*

<!-- project page and paper -->
<!-- [Project Page](https://dydeblur.github.io/) &nbsp; [Paper](https://arxiv.org/abs/2510.10691)  -->
[Paper](https://arxiv.org/abs/2603.27516)

<!-- pageviews -->
<!-- <a href="https://info.flagcounter.com/dhPB"><img src="https://s01.flagcounter.com/mini/dhPB/bg_FFFFFF/txt_000000/border_FFFFFF/flags_0/" alt="Flag Counter" border="0"></a> -->

<!-- teaser -->
![curve](asset/teaser.svg)

Our method achieves high-quality scene-level disentanglement of illumination and material properties from sparse-view input images

## Method Overview
![workflow](asset/pipeline.svg)

<!-- Our method's overall workflow. Dotted arrows and dashed arrows describe the pipeline for modeling camera motion blur and modeling defocus blur, respectively at training time. Solid arrows show the process of rendering sharp images at the inference time. Please refer to the paper for more details. -->

## Todo
<!-- - [ ] ~~Release Paper, Example Code~~ -->
- [x] ~~Release Paper~~ 
- [ ] Release Code (Coming soon)
- [ ] Clean Code

<!-- ## Setup
###  1. Installation
```
git clone https://github.com/hhhddddddd/dydeblur.git --recursive 
cd dydeblur

conda create -n dydeblur python=3.10
conda activate dydeblur

# install pytorch
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y

# install dependencies
pip install -r requirements.txt
```

### 2. Training
```
python train.py -s <dataset> -m <output> -o <expname> -c 0.01 --eval --iterations 40000
```

### 3. Evaluation
```
python render.py -m <output> -o <expname> -c 0.01 -t <time> --mode render 
```
<!-- ![](https://komarev.com/ghpvc/?username=hhhddddddd&color=green) -->



