import os
import math
import timm
import torch
import json
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
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

log_theta = torch.nn.LogSigmoid()


def analyze_neural_activity(args, decoders, sample_features, epoch, output_dir, exp_name, device):
    """
    Analizza l'attività neurale dei decoder (normalizing flows) e crea visualizzazioni
    per monitorare il comportamento dei neuroni durante il training.
    
    Args:
        args: Argomenti di configurazione
        decoders: Lista dei decoder (normalizing flows)
        sample_features: Features estratte dal campione per l'analisi
        epoch: Epoca corrente
        output_dir: Directory di output
        exp_name: Nome esperimento
        device: Device (cuda/cpu)
    """
    
    # Crea directory per salvare le analisi neurali
    neural_dir = Path(output_dir) / exp_name / "neural_analysis"
    neural_dir.mkdir(parents=True, exist_ok=True)
    
    # Directory specifica per questa epoca
    epoch_dir = neural_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(exist_ok=True)
    
    # Inizializza dizionari per raccogliere statistiche
    neuron_stats = defaultdict(list)
    activation_patterns = {}
    
    # Imposta i decoder in modalità eval per l'analisi
    for decoder in decoders:
        decoder.eval()
    
    with torch.no_grad():
        # Analizza ogni livello di feature
        for level_idx, (decoder, features) in enumerate(zip(decoders, sample_features)):
            print(f"Analyzing level {level_idx + 1}/{len(decoders)}...")
            
            # Processa le features in batch per evitare problemi di memoria
            batch_size = min(32, features.shape[0])
            level_activations = []
            level_outputs = []
            
            for batch_start in range(0, features.shape[0], batch_size):
                batch_end = min(batch_start + batch_size, features.shape[0])
                batch_features = features[batch_start:batch_end].to(device)
                
                # Hook per catturare le attivazioni intermedie
                activations = {}
                
                def create_hook(name):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            # Per normalizing flows che restituiscono (z, log_jac_det)
                            activations[name] = output[0].detach().cpu()
                        else:
                            activations[name] = output.detach().cpu()
                    return hook
                
                # Registra gli hook sui layer del decoder
                hooks = []
                for name, module in decoder.named_modules():
                    if len(list(module.children())) == 0:  # Solo layer terminali
                        if any(layer_type in name.lower() for layer_type in ['linear', 'conv', 'norm', 'activation']):
                            hook = module.register_forward_hook(create_hook(name))
                            hooks.append(hook)
                
                # Forward pass attraverso il decoder
                if hasattr(args, 'flow_arch') and args.flow_arch == 'flow_model':
                    z, log_jac_det = decoder(batch_features)
                else:
                    # Assumiamo che serva anche positional encoding
                    bs, dim = batch_features.shape
                    # Crea un positional encoding dummy per l'analisi
                    pos_embed = torch.randn(bs, 128).to(device)  # Dimensione tipica
                    z, log_jac_det = decoder(batch_features, [pos_embed])
                
                # Rimuovi gli hook
                for hook in hooks:
                    hook.remove()
                
                # Raccogli le attivazioni
                for name, activation in activations.items():
                    if activation.numel() > 0:  # Solo se ci sono attivazioni
                        level_activations.append({
                            'name': name,
                            'activation': activation,
                            'batch_idx': batch_start // batch_size
                        })
                
                level_outputs.append(z.detach().cpu())
            
            # Combina tutti gli output del livello
            if level_outputs:
                level_output = torch.cat(level_outputs, dim=0)
                
                # === ANALISI 1: Distribuzione delle attivazioni ===
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle(f'Neural Activity Analysis - Level {level_idx + 1} - Epoch {epoch}', fontsize=16)
                
                # Plot 1: Distribuzione dell'output del decoder
                axes[0, 0].hist(level_output.flatten().numpy(), bins=50, alpha=0.7, color='blue')
                axes[0, 0].set_title(f'Output Distribution (Level {level_idx + 1})')
                axes[0, 0].set_xlabel('Activation Value')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Plot 2: Neuroni attivi (soglia basata su percentile)
                threshold = np.percentile(np.abs(level_output.numpy()), 95)
                active_neurons = (np.abs(level_output.numpy()) > threshold).sum(axis=0)
                axes[0, 1].bar(range(len(active_neurons)), active_neurons, color='green', alpha=0.7)
                axes[0, 1].set_title(f'Active Neurons per Dimension (threshold={threshold:.3f})')
                axes[0, 1].set_xlabel('Neuron Index')
                axes[0, 1].set_ylabel('Active Count')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Plot 3: Heatmap delle attivazioni (campione)
                sample_size = min(50, level_output.shape[0])
                sample_indices = np.random.choice(level_output.shape[0], sample_size, replace=False)
                heatmap_data = level_output[sample_indices].numpy()
                
                im = axes[1, 0].imshow(heatmap_data, aspect='auto', cmap='RdBu_r', interpolation='nearest')
                axes[1, 0].set_title(f'Activation Heatmap (Sample)')
                axes[1, 0].set_xlabel('Neuron Index')
                axes[1, 0].set_ylabel('Sample Index')
                plt.colorbar(im, ax=axes[1, 0])
                
                # Plot 4: Statistiche per neurone
                mean_activations = level_output.mean(dim=0).numpy()
                std_activations = level_output.std(dim=0).numpy()
                
                axes[1, 1].scatter(mean_activations, std_activations, alpha=0.6, color='red')
                axes[1, 1].set_title('Mean vs Std per Neuron')
                axes[1, 1].set_xlabel('Mean Activation')
                axes[1, 1].set_ylabel('Std Activation')
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(epoch_dir / f'neural_activity_level_{level_idx + 1}.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # === ANALISI 2: Tracking dei neuroni nel tempo ===
                # Salva statistiche per il tracking temporale
                neuron_stats[f'level_{level_idx + 1}'].append({
                    'epoch': epoch,
                    'mean_activation': mean_activations.mean(),
                    'std_activation': std_activations.mean(),
                    'active_neurons': (np.abs(mean_activations) > threshold).sum(),
                    'total_neurons': len(mean_activations),
                    'sparsity': (np.abs(mean_activations) < 1e-3).sum() / len(mean_activations)
                })
                
                # Salva pattern di attivazione per analisi successive
                activation_patterns[f'level_{level_idx + 1}'] = {
                    'mean': mean_activations,
                    'std': std_activations,
                    'active_count': active_neurons
                }
    
    # === ANALISI 3: Confronto tra livelli ===
    if len(activation_patterns) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Cross-Level Neural Analysis - Epoch {epoch}', fontsize=16)
        
        levels = list(activation_patterns.keys())
        
        # Plot 1: Numero di neuroni attivi per livello
        active_counts = [activation_patterns[level]['active_count'].sum() for level in levels]
        axes[0, 0].bar(range(len(levels)), active_counts, color='purple', alpha=0.7)
        axes[0, 0].set_title('Total Active Neurons per Level')
        axes[0, 0].set_xlabel('Level')
        axes[0, 0].set_ylabel('Active Neurons')
        axes[0, 0].set_xticks(range(len(levels)))
        axes[0, 0].set_xticklabels([f'L{i+1}' for i in range(len(levels))])
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Sparsity per livello
        sparsities = []
        for level in levels:
            mean_acts = activation_patterns[level]['mean']
            sparsity = (np.abs(mean_acts) < 1e-3).sum() / len(mean_acts)
            sparsities.append(sparsity)
        
        axes[0, 1].plot(range(len(levels)), sparsities, 'o-', color='orange', linewidth=2)
        axes[0, 1].set_title('Sparsity per Level')
        axes[0, 1].set_xlabel('Level')
        axes[0, 1].set_ylabel('Sparsity Ratio')
        axes[0, 1].set_xticks(range(len(levels)))
        axes[0, 1].set_xticklabels([f'L{i+1}' for i in range(len(levels))])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Distribuzione delle medie per livello
        for i, level in enumerate(levels):
            mean_acts = activation_patterns[level]['mean']
            axes[1, 0].hist(mean_acts, bins=30, alpha=0.6, label=f'Level {i+1}')
        axes[1, 0].set_title('Mean Activation Distributions')
        axes[1, 0].set_xlabel('Mean Activation')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Correlazione tra livelli (se possibile)
        if len(levels) == 2:
            level1_means = activation_patterns[levels[0]]['mean']
            level2_means = activation_patterns[levels[1]]['mean']
            min_len = min(len(level1_means), len(level2_means))
            
            axes[1, 1].scatter(level1_means[:min_len], level2_means[:min_len], alpha=0.6)
            axes[1, 1].set_title(f'Level 1 vs Level 2 Correlations')
            axes[1, 1].set_xlabel('Level 1 Mean Activation')
            axes[1, 1].set_ylabel('Level 2 Mean Activation')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Correlation plot\navailable for 2 levels', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Level Correlations')
        
        plt.tight_layout()
        plt.savefig(epoch_dir / 'cross_level_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # === SALVATAGGIO DATI ===
    # Salva i dati numerici per analisi successive
    np.save(epoch_dir / 'activation_patterns.npy', activation_patterns)
    
    # Salva un summary testuale
    with open(epoch_dir / 'analysis_summary.txt', 'w') as f:
        f.write(f"Neural Activity Analysis - Epoch {epoch}\n")
        f.write("=" * 50 + "\n\n")
        
        for level_name, pattern in activation_patterns.items():
            f.write(f"{level_name.upper()}:\n")
            f.write(f"  - Total neurons: {len(pattern['mean'])}\n")
            f.write(f"  - Active neurons: {pattern['active_count'].sum()}\n")
            f.write(f"  - Mean activation: {pattern['mean'].mean():.6f}\n")
            f.write(f"  - Std activation: {pattern['std'].mean():.6f}\n")
            f.write(f"  - Sparsity: {(np.abs(pattern['mean']) < 1e-3).sum() / len(pattern['mean']):.3f}\n")
            f.write("\n")
    
    print(f"Neural analysis completed for epoch {epoch}. Saved to: {epoch_dir}")
    
    return neuron_stats, activation_patterns


def extract_sample_features(args, encoder, test_loader, n_samples=64):
    """
    Estrae un campione di features per l'analisi neurale.
    
    Args:
        args: Argomenti di configurazione
        encoder: Encoder per estrarre le features
        test_loader: DataLoader del test set
        n_samples: Numero di campioni da estrarre
    
    Returns:
        Lista di features per ogni livello
    """
    encoder.eval()
    sample_features = [[] for _ in range(args.feature_levels)]
    samples_collected = 0
    
    with torch.no_grad():
        for batch_data in test_loader:
            if samples_collected >= n_samples:
                break
                
            # Estrai immagini dal batch
            if len(batch_data) == 5:  # (image, label, mask, file_name, img_type)
                images = batch_data[0]
            else:  # Formato diverso
                images = batch_data[0]
            
            images = images.to(args.device)
            
            # Estrai features
            features = encoder(images)
            
            # Raccogli features per ogni livello
            for level_idx in range(args.feature_levels):
                feature = features[level_idx]
                bs, dim, h, w = feature.size()
                # Reshape per l'analisi
                feature_reshaped = feature.permute(0, 2, 3, 1).reshape(-1, dim)
                sample_features[level_idx].append(feature_reshaped.cpu())
            
            samples_collected += images.shape[0]
    
    # Concatena tutte le features raccolte
    final_features = []
    for level_idx in range(args.feature_levels):
        if sample_features[level_idx]:
            level_features = torch.cat(sample_features[level_idx], dim=0)
            # Limita al numero richiesto di campioni
            if level_features.shape[0] > n_samples:
                indices = torch.randperm(level_features.shape[0])[:n_samples]
                level_features = level_features[indices]
            final_features.append(level_features)
        else:
            # Crea features dummy se non ci sono dati
            final_features.append(torch.randn(n_samples, 256))  # Dimensione tipica
    
    return final_features