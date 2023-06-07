# FedCL: Federated Contrastive Learning for Multi-center Medical Image Classification  
This is the code for paper [FedCL: Federated Contrastive Learning for Multi-center Medical Image Classification](https://doi.org/10.1016/j.patcog.2023.109739).  

**Abstract:** Federated learning, which allows distributed medical institutions to train a shared deep learning model with privacy protection, has become increasingly popular recently. However, in practical application, due to data heterogeneity between different hospitals, the performance of the model will be degraded in the training process. In this paper, we propose a federated contrastive learning (FedCL) approach. FedCL integrates the idea of contrastive learning into the federated learning framework. Specifically, it combines the local model and the global model for contrastive learning, so that the local model gradually approaches the global model with the increase of communication rounds, which improves the generalization ability of the model. We validate our method on two public datasets. Extensive experiments show that our method is superior to other federated learning algorithms in medical image classification.

**Keywords:** Federated Learning, Contrastive Learning, Image Classification.  
## Dependencies
+ python >= 3.6.13
+ PyTorch >= 1.8.1
+ torchvision >= 0.8.2
## Parameters
|Parameter|Description|
|:----:|:----|
|model|neural network used in training.|
|batch_size|batch_size per gpu.|
|drop_rate|dropout rate.|
|base_lr|maximum epoch number to train.|
|seed|random seed.|
|gpu|GPU to use.|
|local_ep|local epoch.|
|num_users|numbers of users.|
|rounds|communication rounds.|
|num_workers|num_workers.|
|mu|the mu parameter for Contrastive loss.|
|temperature|the temperature parameter for contrastive loss.|
|out_dim|the output dimension for the projection layer.|
## Datasets
You can download the dataset for Task 1: Skin disease classification in [here](https://challenge.isic-archive.com/data/#2018) and Task 2: COVID-19 detection in [here](https://www.kaggle.com/datasets/praveengovi/coronahack-chest-xraydataset).
## Usage
Here is an example to run the model FedCL.
```
python train_main.py --model=densenet121 \
  --batch_size=8 \
  --base_lr=1e-4 \
  --local_ep=1 \
  --num_users=1 \
  --rounds=200 \
  --num_workers=8 \
  --mu=1 \
  --temperature=0.5 \
```
## Citation
Please cite our paper if you find this code useful for your research.
```
@article{Wu2023FedCLFC,
  title={FedCL: Federated Contrastive Learning for Multi-center Medical Image Classification},
  author={Zhenbing Liu and Fengfeng Wu and Yumeng Wang and Mengyu Yang and Xipeng Pan},
  journal={Pattern Recognition},
  year={2023}
}
```
