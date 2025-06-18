import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import numpy as np
from sklearn.metrics import roc_auc_score


class EncoderFineTuner:
    """
    Fine-tuning dell'encoder usando triplet loss con margin adattivo
    """
    
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        # Parametri di fine-tuning
        self.finetune_epochs = getattr(args, 'finetune_epochs', 10)
        self.finetune_lr = getattr(args, 'finetune_lr', 1e-5)
        self.triplet_margin_alpha = getattr(args, 'triplet_margin_alpha', 1.5)
        self.gradient_clip_norm = getattr(args, 'gradient_clip_norm', 1.0)
        self.freeze_early_layers = getattr(args, 'freeze_early_layers', True)
        self.freeze_ratio = getattr(args, 'freeze_ratio', 0.3)  # Congela il 50% dei primi layer
        
        # Logging
        self.log_file = os.path.join(args.output_dir, args.exp_name, "encoder_finetune_log.csv")
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup del file di logging"""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'triplet_loss', 'margin', 'learning_rate'])
    
    def _freeze_early_layers(self, encoder):
        """
        Congela i primi layer dell'encoder per stabilità
        """
        if not self.freeze_early_layers:
            return
        
        # Ottieni tutti i parametri
        all_params = list(encoder.parameters())
        freeze_count = int(len(all_params) * self.freeze_ratio)
        
        # Congela i primi layer
        for i, param in enumerate(all_params):
            if i < freeze_count:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        print(f"[FINETUNE] Frozen {freeze_count}/{len(all_params)} parameters")
    
    def _compute_triplet_loss(self, features, masks):
        """
        Calcola la triplet loss con margin adattivo e hard negative mining
        
        Args:
            features: Feature estratte dall'encoder
            masks: Maschere di anomalia
        
        Returns:
            triplet_loss, margin_value
        """
        final_features = features[-1]  # Ultimo livello
        bs, dim, h, w = final_features.size()
        
        mask_resized = F.interpolate(masks, size=(h, w), mode='nearest').squeeze(1)
        mask_flat = mask_resized.reshape(-1)
        features_flat = final_features.permute(0, 2, 3, 1).reshape(-1, dim)
        
        # Normalizzazione opzionale
        features_norm = F.normalize(features_flat, dim=1)
        # features_norm = features_flat  # Uncommenta questa per disabilitare normalizzazione

        normal_features = features_norm[mask_flat == 0]
        anomaly_features = features_norm[mask_flat == 1]

        if len(normal_features) < 2 or len(anomaly_features) < 1:
            return torch.tensor(0.0, device=self.device), 0.0

        n_triplets = min(64, len(normal_features) // 2)
        total_loss = 0.0
        total_margin = 0.0

        for _ in range(n_triplets):
            # Anchor e positive da normali
            normal_indices = torch.randperm(len(normal_features))[:2]
            anchor = normal_features[normal_indices[0]].unsqueeze(0)
            positive = normal_features[normal_indices[1]].unsqueeze(0)

            # HARD NEGATIVE MINING: anomalia più vicina all'anchor
            dists = torch.cdist(anchor, anomaly_features, p=2).squeeze(0)
            closest_idx = torch.argmin(dists)
            negative = anomaly_features[closest_idx].unsqueeze(0)

            # Margin dinamico
            anchor_positive_dist = F.pairwise_distance(anchor, positive)
            dynamic_margin = self.triplet_margin_alpha * anchor_positive_dist.mean()

            triplet_loss = F.triplet_margin_loss(
                anchor, positive, negative,
                margin=dynamic_margin.item(),
                p=2, reduction='mean'
            )

            total_loss += triplet_loss
            total_margin += dynamic_margin.item()

        avg_loss = total_loss / n_triplets if n_triplets > 0 else torch.tensor(0.0, device=self.device)
        avg_margin = total_margin / n_triplets if n_triplets > 0 else 0.0

        return avg_loss, avg_margin
    def _validate_encoder(self, encoder, test_loader):
        """
        Validazione rapida dell'encoder fine-tuned
        """
        encoder.eval()
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                if batch_idx >= 10:  # Valida solo su pochi batch per velocità
                    break
                
                if len(data) == 5:
                    image, label, mask, _, _ = data
                else:
                    image, label, mask = data
                
                image = image.to(self.device)
                features = encoder(image)
                
                # Usa le feature dell'ultimo livello per la valutazione
                final_features = features[-1]
                pooled_features = F.adaptive_avg_pool2d(final_features, (1, 1)).flatten(1)
                
                all_features.append(pooled_features.cpu().numpy())
                all_labels.extend(label.numpy())
        
        if len(all_features) == 0:
            return 0.0
        
        # Calcola una metrica semplice (distanza dalle features normali)
        features_array = np.concatenate(all_features, axis=0)
        labels_array = np.array(all_labels)
        
        # Calcola centroide delle features normali
        normal_features = features_array[labels_array == 0]
        if len(normal_features) > 0:
            normal_centroid = np.mean(normal_features, axis=0)
            distances = np.linalg.norm(features_array - normal_centroid, axis=1)
            
            if len(np.unique(labels_array)) > 1:
                auc = roc_auc_score(labels_array, distances)
                return auc
        
        return 0.0
    
    def finetune_encoder(self, encoder, train_loader, test_loader=None):
        """
        Fine-tuning principale dell'encoder
        
        Args:
            encoder: Modello encoder da fine-tunare
            train_loader: DataLoader per il training
            test_loader: DataLoader per la validazione (opzionale)
        
        Returns:
            encoder: Encoder fine-tuned e in modalità eval
        """
        print(f"[FINETUNE] Starting encoder fine-tuning for {self.finetune_epochs} epochs")
        

        # Prepara l'encoder per il fine-tuning
        encoder.train()
        self._freeze_early_layers(encoder)
        
        # Setup ottimizzatore
        trainable_params = [p for p in encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.finetune_lr, weight_decay=1e-4)
        
        # Scheduler per learning rate
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.finetune_epochs, eta_min=self.finetune_lr * 0.1
        )
        
        # Training loop
        for epoch in range(self.finetune_epochs):
            epoch_loss = 0.0
            epoch_margin = 0.0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{self.finetune_epochs}")
            
            for batch_data in progress_bar:
                # Parse batch data
                if len(batch_data) == 5:
                    image, _, mask, _, _ = batch_data
                else:
                    image, _, mask = batch_data
                
                image = image.to(self.device)
                mask = mask.to(self.device)
                
                # Forward pass
                features = encoder(image)
                
                # Calcola triplet loss
                triplet_loss, margin = self._compute_triplet_loss(features, mask)
                
                if triplet_loss.item() > 0:  # Solo se abbiamo una loss valida
                    # Backward pass
                    optimizer.zero_grad()
                    triplet_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.gradient_clip_norm)
                    
                    optimizer.step()
                    
                    epoch_loss += triplet_loss.item()
                    epoch_margin += margin
                    num_batches += 1
                
                # Update progress bar
                current_lr = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix({
                    'Loss': f'{triplet_loss.item():.4f}',
                    'Margin': f'{margin:.4f}',
                    'LR': f'{current_lr:.2e}'
                })
            
            # Update learning rate
            scheduler.step()
            
            # Calcola metriche epoch
            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                avg_margin = epoch_margin / num_batches
                current_lr = optimizer.param_groups[0]['lr']
                
                # Log delle metriche
                with open(self.log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, avg_loss, avg_margin, current_lr])
                
                print(f"[FINETUNE] Epoch {epoch+1}: Loss={avg_loss:.4f}, Margin={avg_margin:.4f}, LR={current_lr:.2e}")
                
                # Validazione opzionale
                if test_loader is not None and (epoch + 1) % 5 == 0:
                    val_auc = self._validate_encoder(encoder, test_loader)
                    print(f"[FINETUNE] Validation AUC: {val_auc:.4f}")
                    encoder.train()  # Torna in modalità training
        
        # Metti l'encoder in modalità eval
        encoder.eval()
        
        # Scongela tutti i parametri per l'uso successivo
        for param in encoder.parameters():
            param.requires_grad = True
        
        print(f"[FINETUNE] Fine-tuning completed. Encoder ready for evaluation.")
        
        return encoder


def finetune_encoder_wrapper(args, encoder, train_loader, test_loader=None):
    """
    Wrapper function per il fine-tuning dell'encoder
    
    Args:
        args: Argomenti di configurazione
        encoder: Encoder da fine-tunare
        train_loader: DataLoader per il training
        test_loader: DataLoader per la validazione (opzionale)
    
    Returns:
        encoder: Encoder fine-tuned e pronto per l'uso
    """
    # Controlla se il fine-tuning è abilitato
    if False:#not getattr(args, 'enable_encoder_finetune', False):
        print("[FINETUNE] Encoder fine-tuning disabled. Returning original encoder.")
        return encoder
    encoder = encoder.to(args.device)
    # Crea il fine-tuner
    fine_tuner = EncoderFineTuner(args)
    
    # Esegui il fine-tuning
    finetuned_encoder = fine_tuner.finetune_encoder(encoder, train_loader, test_loader)
    
    return finetuned_encoder