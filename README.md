Paper link: https://ieeexplore.ieee.org/document/9989417

<p align="left">
    <img src='https://img.shields.io/badge/Paper-MHGCF-blue.svg' alt="Build Status">
</p>
<p align="left">
    <img src='https://img.shields.io/badge/key word-Recommender Systems-green.svg' alt="Build Status">
    <img src='https://img.shields.io/badge/key word-Graph Neural Networks-green.svg' alt="Build Status">
    <img src='https://img.shields.io/badge/key word-Multimodal user preferences-green.svg' alt="Build Status">
</p>

![framework of MHGCF](model.png)


Multimedia-based recommendation (MMRec) is a challenging task, which goes beyond the collaborative filtering (CF) schema that only captures collaborative signals from interactions and explores multimodal user preference cues hidden in complex multimedia content. Despite the significant progress of current solutions for MMRec, we argue that they are limited by multimodal noise contamination. Specifically, a considerable amount of preference-irrelevant multimodal noise (e.g., the background, layout, and brightness in the image of the product) is incorporated into the representation learning of items, which contaminates the modeling of multimodal user preferences. Moreover, most of the latest researches are based on graph convolution networks (GCNs), which means that multimodal noise contamination is further amplified because noisy information is continuously propagated over the user–item interaction graph as recursive neighbor aggregations are performed. 

To address this problem, instead of the common MMRec paradigm which learns user preferences in an integrated manner, we propose a hierarchical framework to separately learn collaborative signals and multimodal preferences cues, thus preventing multimodal noise from flowing into collaborative signals. Then, to alleviate the noise contamination for multimodal user preference modeling, we propose to extract semantic entities from multimodal content that are more relevant to user interests, which can model semantic-level multimodal preferences and thus remove a large fraction of noise. Furthermore, we use the full multimodal features to model content-level multimodal preferences like the existing MMRec solutions, which ensures the sufficient utilization of multimodal information. Overall, we develop a novel model, **m**ultimodal **h**ierarchical **g**raph **CF** (**MHGCF**), which consists of three types of GCN modules tailored to capture collaborative signals, semantic-level preferences, and content-level preferences, respectively. We conduct extensive experiments to demonstrate the effectiveness of MHGCF and its components. 


We provide tensorflow implementation for MHGCF. 

### Before running the codes, please download the [**datasets**](https://www.aliyundrive.com/s/BSZuTyLWT4Y) and copy them to the Data directory.

## prerequisites

- Tensorflow 1.10.0
- Python 3.5
- NVIDIA GPU + CUDA + CuDNN

## Results Reproduction
We reran our code and the results were recorded in the **Model/Log/** directory. Here are the optimal results for MHGCF on the three datasets:
- **Amazon-beauty**: Epoch 249 [8.0s + 65.8s]: train==[16.92742=2.36366 + 4.61919],hit@5=[0.56831],,hit@10=[0.67026],hit@20=[0.76258],ndcg@5=[0.44566],ndcg@10=[0.47886],ndcg@20=[0.50213]
- **Art**: Epoch 199 [11.0s + 52.5s]: train==[19.34818=3.47379 + 4.86527],hit@5=[0.71739],,hit@10=[0.80004],hit@20=[0.87165],ndcg@5=[0.62010],ndcg@10=[0.64696],ndcg@20=[0.66513]
- **Taobao**: Epoch 114 [4.1s + 56.8s]: train==[8.20427=1.40845 + 3.73030],hit@5=[0.41845],,hit@10=[0.52835],hit@20=[0.66265],ndcg@5=[0.31469],ndcg@10=[0.35017],ndcg@20=[0.38390]

## Citation :satisfied:
If our paper and codes are useful to you, please cite:

@ARTICLE{MHGCF,  
author={Liu, Kang and Xue, Feng and Li, Shuaiyang and Sang, Sheng and Hong, Richang},  
journal={IEEE Transactions on Computational Social Systems},   
title={Multimodal Hierarchical Graph Collaborative Filtering for Multimedia-Based Recommendation},   
year={2022},  
pages={1--12},  
doi={10.1109/TCSS.2022.3226862}  
}

@ARTICLE{MEGCF,  
author={Liu, Kang and Xue, Feng and Guo, Dan and Wu, Le and Li, Shujie and Hong, Richang},  
journal={ACM Transactions on Information Systems},   
title={{MEGCF}: Multimodal Entity Graph Collaborative Filtering for Personalized Recommendation},   
year={2022},  
pages={1--26}
}

@ARTICLE{jmpgcf,  
author={Liu, Kang and Xue, Feng and He, Xiangnan and Guo, Dan and Hong, Richang},  
journal={IEEE Transactions on Computational Social Systems},   
title={Joint Multi-Grained Popularity-Aware Graph Convolution Collaborative Filtering for Recommendation},   
year={2022},  
pages={1--12},  
doi={10.1109/TCSS.2022.3151822}
}
