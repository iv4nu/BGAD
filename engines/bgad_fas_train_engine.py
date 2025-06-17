import os
import math
import timm
import torch
import json
import csv
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from utils import t2np, get_logp, adjust_learning_rate, warmup_learning_rate, save_results, save_weights, load_weights
from datasets import create_fas_data_loader,create_test_data_loader
from models import positionalencoding2d, load_flow_model
from losses import get_logp_boundary, calculate_bg_spp_loss, normal_fl_weighting, abnormal_fl_weighting
from utils.visualizer import plot_visualizing_results
from utils.utils import MetricRecorder, calculate_pro_metric, convert_to_anomaly_scores, evaluate_thresholds
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from PIL import Image
from engines.monitoring import analyze_neural_activity,extract_sample_features
from .encoder_train import finetune_encoder_wrapper

log_theta = torch.nn.LogSigmoid()


def train_meta_epoch(args, epoch, data_loader, encoder, decoders, optimizer):
    N_batch = 4096
    decoders = [decoder.train() for decoder in decoders]  # 3
    adjust_learning_rate(args, optimizer, epoch)
    I = len(data_loader)
    
    #mod3
     # CSV log file per triplet loss
    triplet_log_file = os.path.join(args.output_dir, args.exp_name, "triplet_loss_log.csv")
    write_triplet_header = not os.path.exists(triplet_log_file)
#mod3 stop

    # First epoch only training on normal samples to keep training steadily
    if epoch == 0:
        data_loader = data_loader[0]
    else:
        data_loader = data_loader[1]
    for sub_epoch in range(args.sub_epochs):
        total_loss, loss_count = 0.0, 0
        for i, (data) in enumerate(tqdm(data_loader)):
            # warm-up learning rate
            lr = warmup_learning_rate(args, epoch, i+sub_epoch*I, I*args.sub_epochs, optimizer)

            if epoch == 0:
                image, _, mask, _, _ = data
            else:
                image, _, mask = data
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

                # ---- Triplet Loss Analitica SOLO livello finale ----
                if l == args.feature_levels - 1:
                    norm_feats = F.normalize(e[mask_ == 0], dim=1)
                    anom_feats = F.normalize(e[mask_ == 1], dim=1)
                    if len(norm_feats) >= 2 and len(anom_feats) >= 1:
                        perm_n = torch.randperm(len(norm_feats))[:2]
                        perm_a = torch.randint(0, len(anom_feats), (1,))
                        anchor = norm_feats[perm_n[0]].unsqueeze(0)
                        positive = norm_feats[perm_n[1]].unsqueeze(0)
                        negative = anom_feats[perm_a]
                        alpha = 1.5
                        dynamic_margin = alpha * F.pairwise_distance(anchor, positive).mean()
                        triplet_loss = F.triplet_margin_loss(anchor, positive, negative, margin=dynamic_margin.item(), p=2)
                        #print(f"[Triplet - Epoch {epoch}] margin: {dynamic_margin.item():.4f}, loss: {triplet_loss.item():.4f}")
                        with open(triplet_log_file, 'a') as f:
                            if write_triplet_header:
                                f.write("epoch,margin,triplet_loss\n")
                                write_triplet_header = False
                            f.write(f"{epoch},{dynamic_margin.item():.4f},{triplet_loss.item():.4f}\n")

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

                    # first epoch only training normal samples
                    if epoch == 0:
                        if m_b.sum() == 0:  # only normal loss
                            logps = get_logp(dim, z, log_jac_det) 
                            logps = logps / dim
                            loss = -log_theta(logps).mean()

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()
                            loss_count += 1
                    else:
                        if m_b.sum() == 0:  # only normal ml loss
                            logps = get_logp(dim, z, log_jac_det)  
                            logps = logps / dim
                            if args.focal_weighting:
                                normal_weights = normal_fl_weighting(logps.detach())
                                loss = -log_theta(logps) * normal_weights
                                loss = loss.mean()
                            else:
                                loss = -log_theta(logps).mean()
                        if m_b.sum() > 0:  # normal ml loss and bg_sppc loss
                            logps = get_logp(dim, z, log_jac_det)  
                            logps = logps / dim 
                            if args.focal_weighting:
                                logps_detach = logps.detach()
                                normal_logps = logps_detach[m_b == 0]
                                anomaly_logps = logps_detach[m_b == 1]
                                nor_weights = normal_fl_weighting(normal_logps)
                                ano_weights = abnormal_fl_weighting(anomaly_logps)
                                weights = nor_weights.new_zeros(logps_detach.shape)
                                weights[m_b == 0] = nor_weights
                                weights[m_b == 1] = ano_weights
                                loss_ml = -log_theta(logps[m_b == 0]) * nor_weights # (256, )
                                loss_ml = torch.mean(loss_ml)
                            else:
                                loss_ml = -log_theta(logps[m_b == 0])
                                loss_ml = torch.mean(loss_ml)

                            #boundaries = get_logp_boundary(logps,m_b,margin_tau=args.margin_tau,normalizer=args.normalizer,adaptive=False,epoch=epoch,warmup_epochs=7)  # oppure args.warmup_epochs se definito
                            boundaries = get_logp_boundary(logps,m_b,margin_tau=args.margin_tau,pos_beta=0.05,normalizer=args.normalizer,epoch=args.current_epoch,adaptive=True )  # oppure args.warmup_epochs se definito
                            #print('feature level: {}, pos_beta: {}, boudaris: {}'.format(l, args.pos_beta, boundaries))
                            if args.focal_weighting:
                                loss_n_con, loss_a_con = calculate_bg_spp_loss(logps, m_b, boundaries, args.normalizer, weights=weights)
                            else:
                                loss_n_con, loss_a_con = calculate_bg_spp_loss(logps, m_b, boundaries, args.normalizer)


                            loss = loss_ml + args.bgspp_lambda * (loss_n_con + loss_a_con)

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

        # mean_loss = total_loss / loss_count
        # print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(epoch, sub_epoch, mean_loss, lr))


def validate(args, epoch, data_loader, encoder, decoders):
    print('\nCompute loss and scores on category: {}'.format(args.class_name))

    decoders = [decoder.eval() for decoder in decoders]

    image_list, gt_label_list, gt_mask_list, file_names, img_types = [], [], [], [], []
    logps_list = [list() for _ in range(args.feature_levels)]
    total_loss, loss_count = 0.0, 0
    with torch.no_grad():
        for i, (image, label, mask, file_name, img_type) in enumerate(tqdm(data_loader)):
            # image: (32, 3, 256); label: (32, ); mask: (32, 1, 256, 256)
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
                
                ''' if args.vis and epoch == args.meta_epochs - 1 and l == args.feature_levels - 1:
                    
                    save_dir = os.path.join(args.output_dir, args.exp_name, "vis_heatmaps")
                    os.makedirs(save_dir, exist_ok=True)

                    count_saved = 0
                    for j in range(bs):
                        if label[j].item() != 1:
                            continue

                        img = image[j].cpu()
                        save_image(img, os.path.join(save_dir, f"img_{count_saved}_original.png"))

                        score = logps[j].cpu().numpy()
                        score = (score - np.min(score)) / (np.max(score) - np.min(score) + 1e-8)
                        plt.imsave(os.path.join(save_dir, f"img_{count_saved}_anomaly.png"), score, cmap='jet')

                        mask_resized = F.interpolate(mask[j:j+1], size=(score.shape[0], score.shape[1]), mode='nearest')[0, 0].cpu().numpy()
                        plt.imsave(os.path.join(save_dir, f"img_{count_saved}_gtmask.png"), mask_resized, cmap='gray')

                        img_pil = TF.to_pil_image(img)
                        heat = cm.get_cmap('jet')(score)
                        heat = np.uint8(255 * heat[..., :3])
                        heat_img = TF.to_pil_image(torch.from_numpy(heat).permute(2, 0, 1))
                        overlay = Image.blend(img_pil.convert('RGBA'), heat_img.convert('RGBA'), alpha=0.5)
                        overlay.save(os.path.join(save_dir, f"img_{count_saved}_overlay.png"))

                        count_saved += 1
                        if count_saved >= 3:
                            break'''

    mean_loss = total_loss / loss_count
    print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, mean_loss))

    scores = convert_to_anomaly_scores(args, logps_list)
    # calculate detection AUROC
    img_scores = np.max(scores, axis=(1, 2))
    #AGGIUNTO PER TEST CON UAD!!!!!!!
    np.save(os.path.join(args.output_dir, args.exp_name, "img_scores.npy"), img_scores)

    gt_label = np.asarray(gt_label_list, dtype=bool)
    img_auc = roc_auc_score(gt_label, img_scores)
    # calculate segmentation AUROC
    gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)
    pix_auc = roc_auc_score(gt_mask.flatten(), scores.flatten())

    #pix_auc = -1
    pix_pro = -1

    args.pro=False
    if args.pro:
        pix_pro = calculate_pro_metric(scores, gt_mask)
    
    args.vis = True  # forza visualizzazione solo nel test finale (facoltativo)

    if args.vis and epoch == args.meta_epochs - 1:
        img_threshold, pix_threshold = evaluate_thresholds(gt_label, gt_mask, img_scores, scores)
        save_dir = os.path.join(args.output_dir, args.exp_name, 'vis_results', args.class_name)
        os.makedirs(save_dir, exist_ok=True)
        plot_visualizing_results(image_list, scores, img_scores, gt_mask_list, pix_threshold, 
                               img_threshold, save_dir, file_names, img_types)

    # --- Calcolo Precision e Recall pixel-level ---
    bin_scores = (scores > 0.5).astype(np.uint8)  # threshold grezza per predizione binaria
    gt_mask_flat = gt_mask.flatten()
    bin_scores_flat = bin_scores.flatten()

    pix_precision = precision_score(gt_mask_flat, bin_scores_flat, zero_division=0)
    pix_recall = recall_score(gt_mask_flat, bin_scores_flat, zero_division=0)
    # Precision e Recall pixel-level
    TP = np.sum((gt_mask_flat == 1) & (bin_scores_flat == 1))
    FP = np.sum((gt_mask_flat == 0) & (bin_scores_flat == 1))
    FN = np.sum((gt_mask_flat == 1) & (bin_scores_flat == 0))
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    # --- Salvataggio su CSV ---
    # Percorso completo del file CSV
    csv_path = os.path.join(args.output_dir, args.exp_name, "metrics_summary.csv")

    # Controlla se il file esiste già
    file_exists = os.path.isfile(csv_path)

    log_dir = os.path.join(args.output_dir, args.exp_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    csv_file = os.path.join(log_dir, f'{args.class_name}_metrics.csv')
    write_header = not os.path.exists(csv_file)



    # Scrivi o aggiungi al file CSV
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Epoch", "Loss", "Precision", "Recall", "IMG_AUROC", "PIX_AUROC"])
        writer.writerow([epoch, round(mean_loss, 4), round(precision, 4), round(recall, 4),
                        round(img_auc * 100, 2), round(pix_auc * 100, 2)])
    with open(csv_file, 'a') as f:
        if write_header:
            f.write("epoch,loss,img_auc,pix_auc,precision,recall\n")
        f.write(f"{epoch},{mean_loss:.4f},{img_auc:.4f},{pix_auc:.4f},{pix_precision:.4f},{pix_recall:.4f}\n")

    return img_auc, pix_auc, pix_pro


def train(args):
    # Feature Extractor
    encoder = timm.create_model(args.backbone_arch, features_only=True, 
                out_indices=[i+1 for i in range(args.feature_levels)], pretrained=True)
    
    #blocco precedente { 
    #encoder = encoder.to(args.device).eval()
    #feat_dims = encoder.feature_info.channels()
#}
    #blocco aggiornato { 
      # data loaders (li creiamo prima per il fine-tuning)
    normal_loader, train_loader, test_loader = create_fas_data_loader(args) 
    
    # 🆕 FINE-TUNING DELL'ENCODER CON TRIPLET LOSS
    encoder = finetune_encoder_wrapper(args, encoder, train_loader, test_loader)
    
    # Ora l'encoder è già in modalità eval e fine-tuned
    feat_dims = encoder.feature_info.channels()
    
    #}'''
    
    encoder = encoder.to(args.device).eval()
    # Normalizing Flows
    decoders = [load_flow_model(args, feat_dim) for feat_dim in feat_dims]
    decoders = [decoder.to(args.device) for decoder in decoders]
    params = list(decoders[0].parameters())
    for l in range(1, args.feature_levels):
        params += list(decoders[l].parameters())
    # optimizer
    optimizer = torch.optim.Adam(params, lr=args.lr)
    # data loaders
    normal_loader, train_loader, test_loader = create_fas_data_loader(args) 

    # stats
    img_auc_obs = MetricRecorder('IMG_AUROC')
    pix_auc_obs = MetricRecorder('PIX_AUROC')
    pix_pro_obs = MetricRecorder('PIX_AUPRO')

    # Creo il dizionario per tenere traccia delle metriche
    metrics_history = {
        'epochs': [],
        'img_auroc': [],
        'pix_auroc': [],
        'pix_pro': []
    }

    # Creo la directory per i log se non esiste
    log_dir = os.path.join(args.output_dir, args.exp_name, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(args.meta_epochs):
        args.current_epoch = epoch  # ⬅️ Per Relazione
        if args.checkpoint:
            load_weights(encoder, decoders, args.checkpoint)

        print('Train meta epoch: {}'.format(epoch))
        train_meta_epoch(args, epoch, [normal_loader, train_loader], encoder, decoders, optimizer)

        img_auc, pix_auc, pix_pro = validate(args, epoch, test_loader, encoder, decoders)

        img_auc_obs.update(100.0 * img_auc, epoch)
        pix_auc_obs.update(100.0 * pix_auc, epoch)
        pix_pro_obs.update(100.0 * pix_pro, epoch)

        # Aggiungo le metriche al dizionario
        metrics_history['epochs'].append(epoch)
        metrics_history['img_auroc'].append(float(100.0 * img_auc))
        metrics_history['pix_auroc'].append(float(100.0 * pix_auc))
        metrics_history['pix_pro'].append(float(100.0 * pix_pro))

        # Salvo il file JSON ad ogni epoca
        log_file = os.path.join(log_dir, f'{args.class_name}_metrics.json')
        with open(log_file, 'w') as f:
            json.dump(metrics_history, f, indent=4)
        # 🆕 ANALISI NEURALE OGNI 5 EPOCHE
        if True:#epoch % 2 == 0 or epoch == args.meta_epochs - 1:
            print(f"\n[NEURAL ANALYSIS] Analyzing neural activity at epoch {epoch}...")
            try:
                # Estrai features di esempio per l'analisi
                sample_features = extract_sample_features(args, encoder, test_loader, n_samples=64)
                
                # Analizza l'attività neurale
                analyze_neural_activity(args,decoders, sample_features, epoch,args.output_dir, args.exp_name, args.device)
                
                print(f"[NEURAL ANALYSIS] Neural activity analysis saved for epoch {epoch}")
            except Exception as e:
                print(f"[NEURAL ANALYSIS] Error during neural analysis: {e}")
    
    
    
    if args.save_results:
        save_results(img_auc_obs, pix_auc_obs, pix_pro_obs, args.output_dir, args.exp_name, args.model_path, args.class_name)
        save_weights(encoder, decoders, args.output_dir, args.exp_name, args.model_path)
        
        # === Validazione finale su test set ===
    
    print("\n[Post-Training Evaluation] Eseguo validazione finale sul test set...")
    #solo il test loader che già lo restituisce create_fas_data_loader
    encoder.eval()
    decoders = [decoder.eval() for decoder in decoders]

    img_auc, pix_auc, pix_pro = validate(args, args.meta_epochs - 1, test_loader, encoder, decoders)

    print(f"[FINAL] {args.class_name} Image AUC: {img_auc * 100:.2f}")
    print(f"[FINAL] {args.class_name} Pixel AUC: {pix_auc * 100:.2f}")
    print(f"[FINAL] {args.class_name} Pixel PRO: {pix_pro * 100:.2f}")
    
    return img_auc, pix_auc, pix_pro
