# FedCL: Federated Contrastive Learning for Multi-center Medical Image Classification  
This is the code for paper FedCL: Federated Contrastive Learning for Multi-center Medical Image Classification.  

**Abstract:** Federated learning, which allows distributed medical institutions to train a shared deep learning model with privacy protection, has become increasingly popular recently. However, in practical application, due to the data difference between different hospitals, the final training model can not achieve the expected effect. In this paper, we propose a federated contrastive learning (FedCL) approach. FedCL is an effective framework that integrates the idea of contrastive learning into federated learning. It uses the feature representation learned by model to carry out contrastive learning, so that the local model gradually approaches the global model with the increase of communication rounds. A large number of experiments show that our method is superior to other federated learning algorithms in medical image classification. At the same time, we also did a lot of ablation experiments to prove the effectiveness of our method.
## Dependencies
+ python >= 3.6.13
+ PyTorch >= 1.8.1
+ torchvision >= 0.8.2
## Parameters
|Parameter|Description|
|----|----|
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
  --temperature=0.5
```
## Citation
