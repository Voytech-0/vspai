# SPAV: Video SPAI

<div align="center";">

[//]: # (**Name Surname<sup>1</sup>, Name Surname<sup>1</sup>)

<sup>1</sup> University of Amsterdam, The Netherlands  

</div>

[//]: # (<p align="center">)

[//]: # (    <img src="docs/overview.svg" alt="Paper Overview" />)

[//]: # (</p>)

**VSPAI is built on top of SPAI, which employs spectral learning to learn the spectral distribution of real 
images under a self-supervised setup. Then, using the spectral 
reconstruction similarity it detects AI-generated images as out-of-distribution 
samples of this learned model.**

Recent video generation models use Latent Diffusion Models (LDMs) for high-fidelity, 
temporally coherent content. Detection methods like DeMamba and UNITE focus on spatial-temporal artifacts, 
while SPAI uses frequency-domain analysis for images. We extend SPAI's approach to videos for improved detection.

SPAI was chosen as a base of the project because of its innovative approach of learning the distribution
of real images, as opposed to AI-generated ones.

Our contribution involves developping three ways of extending SPAI to videos, fine-tuning them and testing on a diverse 
dataset covering vast selection of generator architectures, resolutions, and scope.

The results of the paper are best summarise by the following table:
# Benchmark Performance

| Model         | Metric | Sora  | CogVideoX | Gen-2 | LaVie | ModelScope | VideoCrafter1 | AVG   |
|---------------|--------|-------|-----------|-------|-------|------------|---------------|-------|
| MViT          | Acc    | 78.6  | **71.2**  | **89.7** | 85.2  | 75.5       | 89.6          | 81.6  |
|               | AP     | **89.2** | **85.2**  | **96.2** | 93.3  | 88.3       | 96.5          | **91.5** |
|               | R      | 68.8  | 54.0      | 91.0   | 82.0  | 62.6       | 90.8          | 74.9  |
| Video Swin    | Acc    | 68.4  | 62.5      | 86.2   | 79.8  | 61.8       | 86.1          | 74.1  |
|               | AP     | 83.8  | 80.7      | 94.8   | 90.5  | 79.8       | 94.4          | 87.4  |
|               | R      | 44.4  | 32.6      | 80.0   | 67.2  | 31.2       | 79.8          | 55.9  |
| VSPAI-1       | Acc    | 64.6  | 54.5      | 81.4   | 89.2  | 77.1       | 93.9          | 78.8  |
|               | AP     | 79.7  | 63.9      | 92.1   | 96.0  | 90.2       | 98.3          | 86.7  |
|               | R      | 34.8  | 14.6      | 68.4   | 84.0  | 59.8       | 93.4          | 59.2  |
| VSPAI-Mean    | Acc    | 73.2  | 57.5      | 87.4   | 85.6  | 79.5       | 93.3          | 79.4  |
|               | AP     | 84.3  | 65.7      | 94.3   | 93.6  | 89.7       | 98.2          | 87.6  |
|               | R      | 55.0  | 23.4      | 83.2   | 79.5  | 67.5       | 95.4          | 67.3  |
| VSPAI-1-FT    | Acc    | 72.1  | 61.2      | 81.6   | 81.8  | 80.4       | 84.0          | 76.8  |
|               | AP     | 80.4  | 64.1      | 92.5   | 96.3  | 90.4       | 98.5          | 87.0  |
|               | R      | 75.8  | 54.0      | 94.8   | 95.2  | 92.4       | 99.6          | 85.3  |
| VSPAI-N (Pool)| Acc    | 77.5  | 65.5      | 82.9   | 81.3  | 80.4       | 83.0          | 78.4  |
|               | AP     | 85.7  | 69.7      | 95.5   | **96.1** | **91.4**   | **98.6**        | 89.5  |
|               | R      | **88.0** | **64.2**  | **98.8** | **95.8** | **94.0**   | **99.2**        | **90.0** |
| VSPAI-N (Mamba)| Acc    | **80.5** | 63.7      | 89.4   | **87.2** | **82.7**   | **91.0**        | **82.4** |
|               | AP     | 86.7  | 71.0      | 95.5   | 94.4  | 90.6       | 97.4          | 89.3  |
|               | R      | 75.2  | 41.6      | 93.0   | 88.6  | 79.8       | 96.2          | 79.1  |

> The table above summarizes the Accuracy \(Acc\), Average Precision \(AP\), and Recall \(R\) scores for each detection method, averaged across four real video sources. The highest values across the options are highlighted in **bold**.

### Conculsion

### Individual Contribution:
- Stipe Frković: Implementing Mamba Architecture
- Izabela Kurek: Testing the developed models
- Lucia Šikulová: Benchmarks
- Wojciech Trejter: Fine-tuning

## :hammer: Installation

[//]: # (### Hardware requirements)


### Required libraries
To train and evaluate VSPAI an anaconda environment can be used for installing all the 
required dependencies as following:

```bash
conda create -n spai python=3.11
conda activate spai
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

### Weights Checkpoint

The trained SPAI weights checkpoint can be downloaded [here](https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view?usp=sharing) 
and should be placed under the `weights` directory, located under the project's root directory.


## :black_nib: License & Contact

This project will download and install additional third-party open 
source software projects. Also, all the employed third-party data 
retain their original license. Review their license terms 
before use.  

The source code and model weights of this project are released under 
the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).
