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

log_theta = torch.nn.LogSigmoid()


def analyze_neural_activity(decoders, sample_data, epoch, output_dir, exp_name, device):
    """
    Analizza l'attività neurale dei decoders e salva grafici informativi
    """
    # Directory per salvare i grafici
    neural_analysis_dir = os.path.join(output_dir, exp_name, "neural_analysis")
    os.makedirs(neural_analysis_dir, exist_ok=True)
    
    # Prepara i dati di esempio per l'analisi
    with torch.no_grad():
        activations_stats = []
        
        for level, decoder in enumerate(decoders):
            decoder.eval()
            
            # Prendi un batch di esempio per l'analisi
            sample_features = sample_data[level]  # Features estratte per questo livello
            
            # Hook per catturare le attivazioni intermedie
            layer_activations = {}
            hooks = []
            
            def create_hook(name):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        layer_activations[name] = output.detach().cpu()
                return hook
            
            # Registra hooks per tutti i layer con attivazione
            layer_count = 0
            for name, module in decoder.named_modules():
                if isinstance(module, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.GELU)):
                    hook = module.register_forward_hook(create_hook(f"level_{level}_layer_{layer_count}_{name}"))
                    hooks.append(hook)
                    layer_count += 1
            
            # Forward pass per catturare le attivazioni
            try:
                _ = decoder(sample_features)
            except:
                # Se fallisce con posizional encoding, prova senza
                pass
            
            # Rimuovi gli hooks
            for hook in hooks:
                hook.remove()
            
            # Analizza le attivazioni catturate
            level_stats = {
                'level': level,
                'dead_neurons_pct': [],
                'mean_activation': [],
                'std_activation': [],
                'layer_names': []
            }
            
            for layer_name, activations in layer_activations.items():
                if activations.numel() > 0:
                    # Calcola statistiche per questo layer
                    flat_activations = activations.flatten()
                    
                    # Percentuale di neuroni "morti" (attivazione sempre <= 0)
                    dead_neurons = (flat_activations <= 0).float().mean().item() * 100
                    
                    # Media e deviazione standard delle attivazioni
                    mean_act = flat_activations.mean().item()
                    std_act = flat_activations.std().item()
                    
                    level_stats['dead_neurons_pct'].append(dead_neurons)
                    level_stats['mean_activation'].append(mean_act)
                    level_stats['std_activation'].append(std_act)
                    level_stats['layer_names'].append(layer_name)
            
            activations_stats.append(level_stats)
        
        # Crea i grafici
        create_neural_activity_plots(activations_stats, epoch, neural_analysis_dir)
        
        # Salva statistiche in JSON
        save_neural_stats(activations_stats, epoch, neural_analysis_dir)


def create_neural_activity_plots(activations_stats, epoch, save_dir):
    """
    Crea grafici informativi sull'attività neurale
    """
    n_levels = len(activations_stats)
    
    # Figura principale con subplot per ogni livello
    fig, axes = plt.subplots(2, n_levels, figsize=(5*n_levels, 10))
    if n_levels == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(f'Neural Activity Analysis - Epoch {epoch}', fontsize=16, fontweight='bold')
    
    for level, stats in enumerate(activations_stats):
        if not stats['layer_names']:  # Skip se non ci sono dati
            continue
            
        # Subplot 1: Percentuale neuroni morti
        ax1 = axes[0, level]
        bars1 = ax1.bar(range(len(stats['dead_neurons_pct'])), stats['dead_neurons_pct'], 
                       color='red', alpha=0.7)
        ax1.set_title(f'Level {level}: Dead Neurons %')
        ax1.set_ylabel('Dead Neurons (%)')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylim(0, 100)
        
        # Aggiungi valori sulle barre
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Subplot 2: Media attivazioni
        ax2 = axes[1, level]
        bars2 = ax2.bar(range(len(stats['mean_activation'])), stats['mean_activation'], 
                       color='blue', alpha=0.7)
        ax2.set_title(f'Level {level}: Mean Activations')
        ax2.set_ylabel('Mean Activation')
        ax2.set_xlabel('Layer Index')
        
        # Aggiungi valori sulle barre
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'neural_activity_epoch_{epoch}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Grafico riassuntivo dell'evoluzione nel tempo (se esistono dati precedenti)
    create_evolution_plot(save_dir, epoch)


def create_evolution_plot(save_dir, current_epoch):
    """
    Crea un grafico dell'evoluzione dell'attività neurale nel tempo
    """
    # Carica dati storici se esistono
    history_file = os.path.join(save_dir, 'neural_history.json')
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        epochs = history['epochs']
        avg_dead_neurons = history['avg_dead_neurons']
        avg_mean_activation = history['avg_mean_activation']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Evoluzione neuroni morti
        ax1.plot(epochs, avg_dead_neurons, 'ro-', linewidth=2, markersize=6)
        ax1.set_title('Evolution of Dead Neurons %')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Average Dead Neurons (%)')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Evoluzione attivazioni medie
        ax2.plot(epochs, avg_mean_activation, 'bo-', linewidth=2, markersize=6)
        ax2.set_title('Evolution of Mean Activations')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Average Mean Activation')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'neural_evolution_epoch_{current_epoch}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()


def save_neural_stats(activations_stats, epoch, save_dir):
    """
    Salva le statistiche neurali in formato JSON per tracking storico
    """
    # Calcola medie globali
    all_dead_neurons = []
    all_mean_activations = []
    
    for stats in activations_stats:
        all_dead_neurons.extend(stats['dead_neurons_pct'])
        all_mean_activations.extend(stats['mean_activation'])
    
    avg_dead_neurons = np.mean(all_dead_neurons) if all_dead_neurons else 0
    avg_mean_activation = np.mean(all_mean_activations) if all_mean_activations else 0
    
    # File per storia completa
    history_file = os.path.join(save_dir, 'neural_history.json')
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)
    else:
        history = {'epochs': [], 'avg_dead_neurons': [], 'avg_mean_activation': [], 'detailed_stats': []}
    
    # Aggiungi dati correnti
    history['epochs'].append(epoch)
    history['avg_dead_neurons'].append(float(avg_dead_neurons))
    history['avg_mean_activation'].append(float(avg_mean_activation))
    history['detailed_stats'].append({
        'epoch': epoch,
        'levels': activations_stats
    })
    
    # Salva storia aggiornata
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    # CSV riassuntivo
    csv_file = os.path.join(save_dir, 'neural_summary.csv')
    write_header = not os.path.exists(csv_file)
    
    with open(csv_file, 'a') as f:
        if write_header:
            f.write("epoch,avg_dead_neurons_pct,avg_mean_activation,num_levels,total_layers\n")
        
        total_layers = sum(len(stats['layer_names']) for stats in activations_stats)
        f.write(f"{epoch},{avg_dead_neurons:.2f},{avg_mean_activation:.4f},{len(activations_stats)},{total_layers}\n")


def extract_sample_features(args, encoder, data_loader, n_samples=32):
    """
    Estrae features di esempio per l'analisi neurale
    """
    encoder.eval()
    sample_features = [[] for _ in range(args.feature_levels)]
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if len(data) == 5:  # train data
                image, _, _, _, _ = data
            else:  # test data
                image, _, _ = data
            
            image = image.to(args.device)
            features = encoder(image)
            
            for l in range(args.feature_levels):
                e = features[l]
                bs, dim, h, w = e.size()
                e = e.permute(0, 2, 3, 1).reshape(-1, dim)
                
                # Prendi solo un subset per l'analisi
                sample_size = min(n_samples, e.size(0))
                sample_features[l] = e[:sample_size]
            
            break  # Prendi solo il primo batch
    
    return sample_features