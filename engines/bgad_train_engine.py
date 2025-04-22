import os
import math
import timm
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils import t2np, get_logp, adjust_learning_rate, warmup_learning_rate, save_results, save_weights, load_weights
from datasets import create_data_loader
from models import positionalencoding2d, load_flow_model
from losses import get_logp_boundary, calculate_bg_spp_loss_normal, normal_fl_weighting
from utils.visualizer import plot_visualizing_results
from utils.utils import MetricRecorder, calculate_pro_metric, convert_to_anomaly_scores, evaluate_thresholds

log_theta = torch.nn.LogSigmoid()


def train_meta_epoch(args, epoch, data_loader, encoder, decoders, optimizer):
    N_batch = 4096
    decoders = [decoder.train() for decoder in decoders]  # 3
    adjust_learning_rate(args, optimizer, epoch)
    I = len(data_loader)

    for sub_epoch in range(args.sub_epochs):
        total_loss, loss_count = 0.0, 0
        for i, (data) in enumerate(tqdm(data_loader)):
            # warm-up learning rate
            lr = warmup_learning_rate(args, epoch, i+sub_epoch*I, I*args.sub_epochs, optimizer)

            image, _, mask, _, _ = data
            image = image.to(args.device)  
            mask = mask.to(args.device)
            with torch.no_grad():
                features = encoder(image)
            for l in range(args.feature_levels):
                e = features[l].detach()  
                bs, dim, h, w = e.size()
                mask_ = F.interpolate(mask, size=(h, w), mode='nearest').squeeze(1)
                mask_ = mask_.reshape(-1)
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                
                # (bs, 128, h, w)
                pos_embed = positionalencoding2d(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
                pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
                decoder = decoders[l]
                
                perm = torch.randperm(bs*h*w).to(args.device)
                num_N_batches = bs*h*w // N_batch
                for i in range(num_N_batches):
                    idx = torch.arange(i*N_batch, (i+1)*N_batch)
                    p_b = pos_embed[perm[idx]]  
                    e_b = e[perm[idx]]  
                    m_b = mask_[perm[idx]]  
                    if args.flow_arch == 'flow_model':
                        z, log_jac_det = decoder(e_b)  
                    else:
                        z, log_jac_det = decoder(e_b, [p_b, ])
                    
                    # first epoch only training normal samples without boundaries
                    if epoch == 0:
                        logps = get_logp(dim, z, log_jac_det) 
                        logps = logps / dim
                        loss = -log_theta(logps).mean()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        loss_count += 1
                    else:
                        logps = get_logp(dim, z, log_jac_det)  
                        logps = logps / dim 
                        if args.focal_weighting:
                            logps_detach = logps.detach()
                            nor_weights = normal_fl_weighting(logps_detach)
                            loss_ml = -log_theta(logps) * nor_weights # (256, )
                            loss_ml = torch.mean(loss_ml)
                        else:
                            loss_ml = -log_theta(logps)
                            loss_ml = torch.mean(loss_ml)
            
                        boundaries = get_logp_boundary(logps, m_b, args.pos_beta, args.margin_tau, args.normalizer)
                        #print('feature level: {}, pos_beta: {}, boudaris: {}'.format(l, args.pos_beta, boundaries))
                        if args.focal_weighting:
                            loss_n_con = calculate_bg_spp_loss_normal(logps, m_b, boundaries, args.normalizer, weights=nor_weights)
                        else:
                            loss_n_con = calculate_bg_spp_loss_normal(logps, m_b, boundaries, args.normalizer)
                    
                        loss = loss_ml + loss_n_con
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        loss_item = loss.item()
                        if math.isnan(loss_item):
                            total_loss += 0.0
                            loss_count += 0
                        else:
                            total_loss += loss.item()
                            loss_count += 1

        mean_loss = total_loss / loss_count
        print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_loss, lr))


def validate(args, epoch, data_loader, encoder, decoders):
    print('\nCompute loss and scores on category: {}'.format(args.class_name))
    
    decoders = [decoder.eval() for decoder in decoders]
    
    image_list, gt_label_list, gt_mask_list, file_names, img_types = [], [], [], [], []
    logps_list = [list() for _ in range(args.feature_levels)]
    total_loss, loss_count = 0.0, 0
    with torch.no_grad():
        for i, (image, label, mask, file_name, img_type) in enumerate(tqdm(data_loader)):
            if args.vis:
                image_list.extend(t2np(image))
                file_names.extend(file_name)
                img_types.extend(img_type)
            gt_label_list.extend(t2np(label))
            gt_mask_list.extend(t2np(mask))
            
            image = image.to(args.device) # single scale
            features = encoder(image)  # BxCxHxW
            for l in range(args.feature_levels):
                e = features[l]  # BxCxHxW
                bs, dim, h, w = e.size()
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
               
                # (bs, 128, h, w)
                pos_embed = positionalencoding2d(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
                pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
                decoder = decoders[l]

                if args.flow_arch == 'flow_model':
                    z, log_jac_det = decoder(e)  
                else:
                    z, log_jac_det = decoder(e, [pos_embed, ])

                logps = get_logp(dim, z, log_jac_det)  
                logps = logps / dim  
                loss = -log_theta(logps).mean() 
                total_loss += loss.item()
                loss_count += 1
                logps_list[l].append(logps.reshape(bs, h, w))
    
    mean_loss = total_loss / loss_count
    print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, mean_loss))
    
    scores = convert_to_anomaly_scores(args, logps_list)
    # calculate detection AUROC
    img_scores = np.max(scores, axis=(1, 2))
    gt_label = np.asarray(gt_label_list, dtype=np.bool)
    img_auc = roc_auc_score(gt_label, img_scores)
    # calculate segmentation AUROC
    gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=np.bool), axis=1)
    pix_auc = roc_auc_score(gt_mask.flatten(), scores.flatten())

    pix_pro = -1
    if args.pro:
        pix_pro = calculate_pro_metric(scores, gt_mask)
    
    if args.vis and epoch == args.meta_epochs - 1:
        img_threshold, pix_threshold = evaluate_thresholds(gt_label, gt_mask, img_scores, scores)
        save_dir = os.path.join(args.output_dir, args.exp_name, 'vis_results', args.class_name)
        os.makedirs(save_dir, exist_ok=True)
        plot_visualizing_results(image_list, scores, img_scores, gt_mask_list, pix_threshold, 
                                 img_threshold, save_dir, file_names, img_types)

    return img_auc, pix_auc, pix_pro


import json
import os


# modificata, consente ogni 10 epoche di stampare andamento della loss, Auroc e Pro score 
def train(args):
    import timm
    import torch
    
    # === Caricamento manuale dei pesi preaddestrati EfficientNet-B6 ===
    checkpoint_path = "/kaggle/input/efficientnet-b6-weights/tf_efficientnet_b6_ns-51548356.pth"
    encoder = timm.create_model("tf_efficientnet_b6_ns", features_only=True, 
                out_indices=[i+1 for i in range(args.feature_levels)], pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    encoder.load_state_dict(state_dict)
    encoder = encoder.to(args.device).eval()

    feat_dims = encoder.feature_info.channels()

    decoders = [load_flow_model(args, feat_dim) for feat_dim in feat_dims]
    decoders = [decoder.to(args.device) for decoder in decoders]
    params = list(decoders[0].parameters())
    for l in range(1, args.feature_levels):
        params += list(decoders[l].parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)
    train_loader, test_loader = create_data_loader(args)

    img_auc_obs = MetricRecorder('IMG_AUROC')
    pix_auc_obs = MetricRecorder('PIX_AUROC')
    pix_pro_obs = MetricRecorder('PIX_AUPRO')

    loss_curve = []
    epoch_metrics = []

    for epoch in range(args.meta_epochs):
        if args.checkpoint:
            load_weights(encoder, decoders, args.checkpoint)

        print(f'Train meta epoch: {epoch}')
        train_meta_epoch(args, epoch, train_loader, encoder, decoders, optimizer)

        # Ogni 10 epoche o ultima
        if (epoch + 1) % 10 == 0 or epoch == args.meta_epochs - 1:
            img_auc, pix_auc, pix_pro = validate(args, epoch, test_loader, encoder, decoders)

            img_auc_obs.update(100.0 * img_auc, epoch)
            pix_auc_obs.update(100.0 * pix_auc, epoch)
            pix_pro_obs.update(100.0 * pix_pro, epoch)

            epoch_metrics.append({
                "epoch": epoch,
                "image_auroc": round(100.0 * img_auc, 2),
                "pixel_auroc": round(100.0 * pix_auc, 2),
                "pro_score": round(100.0 * pix_pro, 2)
            })

            # Salvataggio log JSON incrementale
            log_save_path = os.path.join("logs", f"{args.class_name}_log.json")
            os.makedirs("logs", exist_ok=True)
            json.dump({
                "epoch_metrics": epoch_metrics,
                "loss_curve": loss_curve,
                "image_auroc": img_auc_obs.max_score,
                "pixel_auroc": pix_auc_obs.max_score,
                "pro_score": pix_pro_obs.max_score
            }, open(log_save_path, "w"), indent=2)

        # salva andamento loss anche se non validi
        loss_curve.append(-1)

    if args.save_results:
        save_results(img_auc_obs, pix_auc_obs, pix_pro_obs, args.output_dir, args.exp_name, args.model_path, args.class_name)
        save_weights(encoder, decoders, args.output_dir, args.exp_name, args.model_path)

    return img_auc_obs.max_score, pix_auc_obs.max_score, pix_pro_obs.max_score
