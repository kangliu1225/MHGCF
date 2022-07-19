# MHGCF
## Multimodal Hierarchical Graph Collaborative Filtering for Multimedia-based Recommendation

![framework of MHGCF](model.png)

Many state-of-the-art multimedia recommender efforts effectively alleviate the issues of sparsity and cold-start via modeling multimodal user preference. The core paradigm of them is based on graph learning techniques, which perform high-order message propagation of multimodal information on  user-item interaction graph, so as to obtain the node representations that contain both interactive- and multimodal-dimension user preferences. However, we argue that such a paradigm is suboptimal because it ignores two problems, including (1) the presence of a large number of preference-independent noisy data in the items' multimodal content, and (2) the propagation of this multimodal noise over the interaction graph contaminates the representations of both interactive- and multimodal-dimension user preferences.

In this work, we aim to reduce the negative effects of multimodal noise and further improve user preference modeling. Towards this end, we develop a multimodal graph contrastive learning (MGCL) approach, which decomposes user preferences into multiple dimensions and performs cross-dimension mutual information maximization, so that user preference modeling over different dimensions can be enhanced with each other. In particular, we first adopt the graph learning approach to generate representations of users and items in the interaction and multimodal dimensions, respectively. Then, we construct an additional contrastive learning task to maximize the consistency between different dimensions. Extensive experiments on three public datasets validate the effectiveness and scalability of the proposed MGCL.

We provide tensorflow implementation for MEGCF. 

### Before running the codes, please download the [**datasets**](https://www.aliyundrive.com/s/cmEeDMecU88) and copy them to the Data directory.

## prerequisites

- Tensorflow 1.10.0
- Python 3.5
- NVIDIA GPU + CUDA + CuDNN

We reran our code and the results were recorded in the **Log** directory. Here are the optimal results for MHGCF on the three datasets:
- **Amazon-beauty**: Epoch 249 [8.0s + 65.8s]: train==[16.92742=2.36366 + 4.61919],hit@5=[0.56831],,hit@10=[0.67026],hit@20=[0.76258],ndcg@5=[0.44566],ndcg@10=[0.47886],ndcg@20=[0.50213]
- **Art**: 
- **Taobao**: Epoch 114 [4.1s + 56.8s]: train==[8.20427=1.40845 + 3.73030],hit@5=[0.41845],,hit@10=[0.52835],hit@20=[0.66265],ndcg@5=[0.31469],ndcg@10=[0.35017],ndcg@20=[0.38390]

