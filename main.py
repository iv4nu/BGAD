import torch
import torch.nn as nn
import os
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from config import parse_args
from engines.bgad_fas_train_engine import train
#from engines.bgad_train_engine import train
from datasets import create_fas_data_loader,create_data_loader




def main():
    args = parse_args()
    


    # Override parametri
    args.class_name = 'capsule'
    args.flow_arch = 'conditional_flow_model'
    args.margin_tau = 0.1
    args.pos_beta = 0.05
    args.data_strategy = '0,1'
    args.num_anomalies = 5

    
    args.crop_size = (args.inp_size, args.inp_size)
    args.img_size = (args.inp_size, args.inp_size)
    args.norm_mean = [0.485, 0.456, 0.406]
    args.norm_std = [0.229, 0.224, 0.225]
    args.save_results = True
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Warmup defaults
    args.lr_warmup_from = 0.0
    args.lr_warmup_to = args.lr

    # Early stopping config
    patience = 5
    best_score = -1
    best_epoch = 0
    stop_counter = 0

   # normal_loader, train_loader, test_loader = create_fas_data_loader(args)

    img_auc, pix_auc, _ = train(args)


if __name__ == '__main__':
    main()
