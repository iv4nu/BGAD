import torch
import torch.nn as nn
import os
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from main import parse_args
from engines.bgad_fas_train_engine import train
from datasets import create_fas_data_loader

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
            permute_soft=True
        )
    return flow

def main():
    args = parse_args()

    # Parametri specifici per test ottimizzato
    args.class_name = 'bottle'
    args.margin_tau = 0.1
    args.pos_beta = 0.05
    args.flow_arch = 'conditional_flow_model'

    # Early stopping config
    patience = 4
    best_score = -1
    best_epoch = 0
    stop_counter = 0

    normal_loader, train_loader, test_loader = create_fas_data_loader(args)

    # Override del decoder con modello ottimizzato
    decoder = build_optimized_flow_model(input_dim=2048, cond_dim=args.pos_embed_dim)
    decoder = decoder.cuda()

    encoder = torch.hub.load('rwightman/gen-efficientnet-pytorch', args.backbone_arch, pretrained=True)
    encoder = encoder.cuda()

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.meta_epochs):
        print(f"\n[Epoch {epoch}] Training...")
        img_auc, pix_auc, _ = train(args, epoch, [normal_loader, train_loader], encoder, [decoder], optimizer)

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
