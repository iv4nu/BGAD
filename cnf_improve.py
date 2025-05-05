import torch
import torch.nn as nn
import os
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from main import parse_args
from engines.bgad_fas_train_engine import train
from datasets import create_fas_data_loader
from adaptive_boundary_hook import AdaptiveBoundaryHook




def build_optimized_flow_model(input_dim, cond_dim, n_layers=4):
    flow = Ff.SequenceINN(input_dim)
    for _ in range(n_layers):
        flow.append(
            Fm.AllInOneBlock,
            cond=0,
            cond_shape=(cond_dim,),
            subnet_constructor=lambda in_dim, out_dim: nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 256),
                nn.LeakyReLU(),
                nn.Linear(256, out_dim)
            ),
            affine_clamping=1.9,
            global_affine_type='SOFTPLUS',
            permute_soft=False
        )
    return flow


def main():
    args = parse_args()
    
    boundary_hook = AdaptiveBoundaryHook(
    alpha=0.2,
    epsilon=0.01,
    max_epsilon=0.05,
    n_bootstrap=100,
    search_epsilon=True,
    log_path = os.path.join("/kaggle/working/BGAD", "adaptive_boundary_log.csv"),
    verbose=True
)


    # Override parametri
    args.class_name = 'wood'
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

    normal_loader, train_loader, test_loader = create_fas_data_loader(args)

    encoder = torch.hub.load('rwightman/gen-efficientnet-pytorch', args.backbone_arch, pretrained=True).cuda()
    decoder = build_optimized_flow_model(input_dim=2048, cond_dim=args.pos_embed_dim).cuda()

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.meta_epochs):
        print(f"\n[Epoch {epoch}] Training...")
        img_auc, pix_auc, _ = train(args,boundary_hook=boundary_hook)

        if img_auc > best_score:
            best_score = img_auc
            best_epoch = epoch
            stop_counter = 0
        else:
            stop_counter += 1

        print(f"[EarlyStopping] img_auc: {img_auc:.2f}, best: {best_score:.2f} @ epoch {best_epoch}, counter: {stop_counter}/{patience}")

        if stop_counter >= patience:
            print("[EarlyStopping] Fermata anticipata del training.")
            break


if __name__ == '__main__':
    main()
