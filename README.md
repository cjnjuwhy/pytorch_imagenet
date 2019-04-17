# PyTorch Image Classification

- PyTorch models for imagenet classification.
- These models can be utilized as pre-training models for other computer vision tasks.
- Training with the nets provided by models/net_factory.py(self-define & in torch models)


## Training results

| Model | Top-1 error (%) | Top-5 error (%) | Links | Notes |
|---|---|---|---|---|
| PVANet | 26.92 | 8.84 | [Model download ](http://api-metakage-4misc.kakao.com/dn/kakaobrain/ian.theman/pytorch_imagenet/pvanet_600epochs.checkpoint.pth.tar) / [Acc. plot](assets/pvanet_acc.png) / [Loss plot](assets/pvanet_loss.png) | 600 epochs |


## References

- PVANet: Lightweight Deep Neural Networks for Real-time Object Detection (https://arxiv.org/abs/1611.08588)
