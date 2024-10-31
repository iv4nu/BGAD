## [Explicit Boundary Guided Semi-Push-Pull Contrastive Learning for Supervised Anomaly Detection](https://arxiv.org/abs/2207.01463)

PyTorch implementation for CVPR2023 paper, Explicit Boundary Guided Semi-Push-Pull Contrastive Learning for Supervised Anomaly Detection.

<img src="./BGAD-framework.jpg" width="800">

---

## Installation
Install all packages with this command:
```
$ python3 -m pip install -U -r requirements.txt
```

## Download Datasets
Please download MVTecAD dataset from [MVTecAD dataset](https://www.mvtec.com/de/unternehmen/forschung/datasets/mvtec-ad/) and BTAD dataset from [BTAD dataset](http://avires.dimi.uniud.it/papers/btad/btad.zip).


## Training
- Run code for training MVTecAD
```
python main.py --flow_arch conditional_flow_model --gpu 0 --data_path /path/to/your/dataset --with_fas --data_strategy 0,1 --num_anomalies 10 --not_in_test --exp_name bgad_fas_10 --focal_weighting --pos_beta 0.01 --margin_tau 0.1
```
- Run code for training BTAD
```
python main.py --flow_arch conditional_flow_model --gpu 0 --dataset btad --data_path /path/to/your/dataset --with_fas --data_strategy 0,1 --num_anomalies 10 --not_in_test --exp_name bgad_fas_10 --focal_weighting --pos_beta 0.01 --margin_tau 0.1
```

## Testing
- Run code for testing
```
python test.py --flow_arch conditional_flow_model --gpu 0 --checkpoint /path/to/output/dir --phase test --pro 
```


## Citation

If you find this repository useful, please consider citing our work:
```
@article{BGAD,
      title={Explicit Boundary Guided Semi-Push-Pull Contrastive Learning for Supervised Anomaly Detection}, 
      author={Xincheng Yao and Ruoqi Li and Jing Zhang and Jun Sun and Chongyang Zhang},
      year={2023},
      booktitle={Conference on Computer Vision and Pattern Recognition 2023},
      url={https://arxiv.org/abs/2207.01463},
      primaryClass={cs.CV}
}
```

## Acknowledgement

This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, the [CFLOW](https://github.com/gudovskiy/cflow-ad) repository and the [FrEIA](https://github.com/VLL-HD/FrEIA) repository.

If you are interested in our work, you can also follow our other works: [PMAD (AAAI2023)](https://github.com/xcyao00/PMAD), [FOD (ICCV2023)](https://github.com/xcyao00/FOD), [HGAD (ECCV2024)](https://github.com/xcyao00/HGAD), [ResAD (NeurIPS2024)](https://github.com/xcyao00/ResAD). Or, you can follow our github page [xcyao00](https://github.com/xcyao00).
