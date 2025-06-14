from torch import nn
import torch
import torch.nn.functional as F
# FrEIA (https://github.com/VLL-HD/FrEIA/)
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import matplotlib.pyplot as plt
import os
from collections import OrderedDict

# Original subnet (mantenuto per compatibilità)
def subnet_fc(dims_in, dims_out, layer_id=None, args=None):
    layers = nn.Sequential(
                nn.Linear(dims_in, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 256),
                nn.LeakyReLU(),
                nn.Linear(256, out_dim)
            )
    return layers

# Enhanced subnet for multi-scale (con dimensioni adattive)
def adaptive_subnet_fc(dims_in, dims_out, complexity_factor=1.0):
    """
    Subnet che si adatta alla complessità richiesta per diverse scale
    complexity_factor: 0.5 per scale semplici, 1.0 per standard, 1.5 per scale complesse
    """
    hidden_dim = max(128, int(256 * complexity_factor))
    
    layers = nn.Sequential(
        nn.Linear(dims_in, hidden_dim),
        nn.LeakyReLU(0.1),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LeakyReLU(0.1),
        nn.Linear(hidden_dim, dims_out)
    )
    return layers

class MultiScaleFeatureExtractor(nn.Module):
    """
    Estrae features a multiple scale da EfficientNet-B6
    """
    def __init__(self, backbone_model):
        super().__init__()
        self.backbone = backbone_model
        
        # Hook points per EfficientNet-B6 (adatta questi indici al tuo modello specifico)
        self.hook_points = {
            'low_level': 2,      # Early features (alta risoluzione, dettagli locali)
            'mid_level': 4,      # Middle features (media risoluzione, pattern intermedi)  
            'high_level': 6,     # Late features (bassa risoluzione, semantica)
            'global': -1         # Final features (globale)
        }
        
        # Storage per features estratte
        self.extracted_features = {}
        self.hooks = []
        
        # Adaptive pooling per normalizzare dimensioni
        self.adaptive_pools = nn.ModuleDict({
            'low_level': nn.AdaptiveAvgPool2d((4, 4)),   # Mantiene più dettagli spaziali
            'mid_level': nn.AdaptiveAvgPool2d((2, 2)),   # Dettagli intermedi
            'high_level': nn.AdaptiveAvgPool2d((1, 1)),  # Features compatte
            'global': nn.AdaptiveAvgPool2d((1, 1))       # Global features
        })
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Registra hooks per estrarre features intermediate"""
        def make_hook(scale_name):
            def hook_fn(module, input, output):
                # Applica adaptive pooling e flattening
                pooled = self.adaptive_pools[scale_name](output)
                self.extracted_features[scale_name] = pooled.flatten(1)
            return hook_fn
        
        # Registra hooks per ogni scala (eccetto global che viene estratta normalmente)
        for scale_name, layer_idx in self.hook_points.items():
            if scale_name != 'global' and layer_idx != -1:
                if hasattr(self.backbone, 'features'):
                    # EfficientNet structure
                    target_layer = self.backbone.features[layer_idx]
                else:
                    # Adatta alla struttura del tuo modello
                    target_layer = list(self.backbone.children())[layer_idx]
                
                hook = target_layer.register_forward_hook(make_hook(scale_name))
                self.hooks.append(hook)
    
    def forward(self, x):
        """Estrae features a tutte le scale"""
        # Reset storage
        self.extracted_features = {}
        
        # Forward pass (questo attiva gli hooks)
        global_output = self.backbone(x)
        
        # Aggiungi global features
        if len(global_output.shape) > 2:
            global_output = self.adaptive_pools['global'](global_output).flatten(1)
        self.extracted_features['global'] = global_output
        
        return self.extracted_features
    
    def cleanup_hooks(self):
        """Rimuovi hooks quando non servono più"""
        for hook in self.hooks:
            hook.remove()

class MultiScaleNormalizingFlows(nn.Module):
    """
    Sistema di Normalizing Flows a multiple scale
    """
    def __init__(self, args, feature_dims):
        super().__init__()
        self.args = args
        self.scales = list(feature_dims.keys())
        
        # Flow separato per ogni scala
        self.flows = nn.ModuleDict()
        
        # Complexity factors per diverse scale
        complexity_factors = {
            'low_level': 1.2,    # Più complesso per dettagli locali
            'mid_level': 1.0,    # Standard
            'high_level': 0.8,   # Meno complesso per features semantiche
            'global': 1.0        # Standard per features globali
        }
        
        for scale_name, input_dim in feature_dims.items():
            print(f'Creating flow for {scale_name}: input_dim={input_dim}')
            
            # Crea flow per questa scala
            flow = Ff.SequenceINN(input_dim)
            
            # Numero di coupling layers adattivo per scala
            n_layers = self._get_optimal_layers(scale_name)
            complexity = complexity_factors.get(scale_name, 1.0)
            
            for k in range(n_layers):
                flow.append(
                    Fm.AllInOneBlock,
                    cond=0,
                    cond_shape=(args.pos_embed_dim,),
                    subnet_constructor=lambda in_dim, out_dim, cf=complexity: 
                        adaptive_subnet_fc(in_dim, out_dim, cf),
                    affine_clamping=args.clamp_alpha if hasattr(args, 'clamp_alpha') else 1.9,
                    global_affine_type='SOFTPLUS',
                    permute_soft=False
                )
            
            self.flows[scale_name] = flow
        
        # Learnable fusion weights
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales)))
        
        # Optional: attention mechanism tra scale
        self.use_attention = getattr(args, 'use_scale_attention', False)
        if self.use_attention:
            self.scale_attention = nn.MultiheadAttention(
                embed_dim=len(self.scales), 
                num_heads=1
            )
    
    def _get_optimal_layers(self, scale_name):
        """Determina numero ottimale di layers per scala"""
        layer_config = {
            'low_level': 6,      # Meno layers per features semplici
            'mid_level': 8,      # Standard
            'high_level': 8,     # Standard
            'global': 8          # Standard (come originale)
        }
        return layer_config.get(scale_name, 8)
    
    def forward(self, multi_scale_features, condition=None):
        """
        Forward pass attraverso tutti i flows
        Gestisce sia input dict (multi-scale) che tensor singolo (backward compatibility)
        """
        scale_outputs = {}
        log_likelihoods = {}
        
        # Controlla se l'input è un tensor singolo (backward compatibility)
        if isinstance(multi_scale_features, torch.Tensor):
            # Modalità backward compatibility: usa solo il flow 'global'
            if 'global' in self.flows:
                features = multi_scale_features
                
                if condition is not None:
                    z, log_jac_det = self.flows['global'](features, c=condition)
                else:
                    z, log_jac_det = self.flows['global'](features)
                
                # Calcola log-likelihood
                log_prob_z = -0.5 * torch.sum(z**2, dim=1)
                log_likelihood = log_prob_z + log_jac_det
                
                scale_outputs['global'] = z
                log_likelihoods['global'] = log_likelihood
                
                return log_likelihood, log_likelihoods, scale_outputs
            else:
                raise ValueError("Single tensor input provided but no 'global' flow available")
        
        # Input è un dizionario (modalità multi-scale)
        elif isinstance(multi_scale_features, dict):
            # Processa ogni scala
            for scale_name in self.scales:
                if scale_name in multi_scale_features:
                    features = multi_scale_features[scale_name]
                    
                    # Forward attraverso il flow di questa scala
                    if condition is not None:
                        z, log_jac_det = self.flows[scale_name](features, c=condition)
                    else:
                        z, log_jac_det = self.flows[scale_name](features)
                    
                    # Calcola log-likelihood
                    log_prob_z = -0.5 * torch.sum(z**2, dim=1)  # Standard normal prior
                    log_likelihood = log_prob_z + log_jac_det
                    
                    scale_outputs[scale_name] = z
                    log_likelihoods[scale_name] = log_likelihood
            
            # Fusion delle log-likelihoods
            fused_likelihood = self._fuse_likelihoods(log_likelihoods)
            
            return fused_likelihood, log_likelihoods, scale_outputs
        
        else:
            raise TypeError(f"multi_scale_features must be either torch.Tensor or dict, got {type(multi_scale_features)}")
    
    def _fuse_likelihoods(self, log_likelihoods):
        """
        Fusione intelligente delle log-likelihoods da diverse scale
        """
        if not log_likelihoods:
            return torch.tensor(0.0)
        
        # Stack log-likelihoods
        ll_stack = torch.stack([log_likelihoods[scale] for scale in self.scales 
                               if scale in log_likelihoods], dim=1)
        
        if self.use_attention:
            # Attention-based fusion
            ll_attended, _ = self.scale_attention(
                ll_stack.unsqueeze(0), ll_stack.unsqueeze(0), ll_stack.unsqueeze(0)
            )
            fused = ll_attended.squeeze(0).mean(dim=1)
        else:
            # Weighted fusion con learned weights
            weights = F.softmax(self.scale_weights[:ll_stack.shape[1]], dim=0)
            fused = torch.sum(ll_stack * weights.unsqueeze(0), dim=1)
        
        return fused
    
    def reverse(self, z_dict, condition=None):
        """
        Reverse pass per generazione (utile per debugging/analysis)
        """
        reconstructed = {}
        for scale_name, z in z_dict.items():
            if scale_name in self.flows:
                if condition is not None:
                    x_recon = self.flows[scale_name](z, c=condition, rev=True)[0]
                else:
                    x_recon = self.flows[scale_name](z, rev=True)[0]
                reconstructed[scale_name] = x_recon
        return reconstructed

class MultiScaleBGAD(nn.Module):
    """
    Integrazione completa del sistema Multi-Scale nel framework BGAD
    """
    def __init__(self, args, encoder_model, feature_extractor_dims):
        super().__init__()
        self.args = args
        
        # Multi-scale feature extractor
        self.feature_extractor = MultiScaleFeatureExtractor(encoder_model)
        
        # Multi-scale normalizing flows
        self.multi_scale_flows = MultiScaleNormalizingFlows(args, feature_extractor_dims)
        
        # Scale-specific boundaries (opzionale)
        self.use_scale_boundaries = getattr(args, 'use_scale_boundaries', False)
        if self.use_scale_boundaries:
            self.scale_boundaries = nn.ModuleDict({
                scale: self._create_boundary_module(scale) 
                for scale in feature_extractor_dims.keys()
            })
    
    def _create_boundary_module(self, scale_name):
        """Crea modulo per boundary specifico della scala"""
        # Placeholder per boundary logic specifica per scala
        return nn.Identity()
    
    def forward(self, x, condition=None):
        """
        Forward pass completo del sistema multi-scale
        """
        # Estrazione features multi-scale
        multi_scale_features = self.feature_extractor(x)
        
        # Processing attraverso normalizing flows
        fused_likelihood, scale_likelihoods, scale_outputs = self.multi_scale_flows(
            multi_scale_features, condition
        )
        
        results = {
            'fused_likelihood': fused_likelihood,
            'scale_likelihoods': scale_likelihoods,
            'scale_outputs': scale_outputs,
            'multi_scale_features': multi_scale_features
        }
        
        return results
    
    def cleanup(self):
        """Cleanup resources"""
        self.feature_extractor.cleanup_hooks()

class BackwardCompatibleMultiScaleFlow(nn.Module):
    """
    Wrapper che fornisce piena backward compatibility
    Si comporta esattamente come il flow originale quando riceve tensor singoli
    """
    def __init__(self, args, in_channels):
        super().__init__()
        self.args = args
        self.use_multi_scale = getattr(args, 'use_multi_scale', False)
        
        if True:
            # Sistema multi-scale
            feature_dims = {
                'low_level': getattr(args, 'low_level_dim', 512),
                'mid_level': getattr(args, 'mid_level_dim', 1024), 
                'high_level': getattr(args, 'high_level_dim', 1536),
                'global': in_channels
            }
            self.flow = MultiScaleNormalizingFlows(args, feature_dims)
        # else:
        #     # Sistema originale
        #     self.flow = build_optimized_flow_model(args, in_channels)
    
    def forward(self, x, c=None, rev=False):
        """
        Interfaccia compatibile con il sistema originale
        x: tensor o dict
        c: condition tensor
        rev: reverse mode (per generazione)
        """
        if self.use_multi_scale:
            if rev:
                # Reverse mode per multi-scale
                if isinstance(x, dict):
                    return self.flow.reverse(x, condition=c)
                else:
                    # Se x è un tensor singolo, trattalo come 'global'
                    z_dict = {'global': x}
                    result = self.flow.reverse(z_dict, condition=c)
                    return result['global'], torch.zeros(x.shape[0])  # dummy jacobian
            else:
                # Forward mode
                fused_likelihood, scale_likelihoods, scale_outputs = self.flow(x, condition=c)
                
                # Per backward compatibility, restituisci nel formato originale
                if isinstance(x, torch.Tensor):
                    # Input era tensor singolo, restituisci come il sistema originale
                    return scale_outputs['global'], fused_likelihood
                else:
                    # Input era dict, restituisci formato esteso
                    return fused_likelihood, scale_likelihoods, scale_outputs
        else:
            # Sistema originale - passa tutto direttamente
            return self.flow(x, c=c, rev=rev)

# Factory function per creare il modello (compatibile con il tuo codice esistente)
def conditional_flow_model(args, in_channels):
    """
    Wrapper per mantenere compatibilità con il codice esistente
    """
    print(f"Creating flow model - Multi-scale: {getattr(args, 'use_multi_scale', False)}")
    return BackwardCompatibleMultiScaleFlow(args, in_channels)

def flow_model(args, in_channels):
    """
    Wrapper per mantenere compatibilità con il codice esistente
    """
    print(f"Creating flow model - Multi-scale: {getattr(args, 'use_multi_scale', False)}")
    return BackwardCompatibleMultiScaleFlow(args, in_channels)

def build_optimized_flow_model(args, input_dim):
    """Mantiene il tuo sistema originale ottimizzato"""
    flow = Ff.SequenceINN(input_dim)
    for _ in range(8):
        flow.append(
            Fm.AllInOneBlock,
            cond=0,
            cond_shape=(args.pos_embed_dim,),
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

# Utility per configurazione semplificata
def setup_multiscale_args(args, enable_multiscale=True):
    """
    Helper per configurare args per multi-scale
    """
    args.use_multi_scale = enable_multiscale
    args.use_scale_attention = False  # Start simple
    args.use_scale_boundaries = False  # Start simple
    
    # Dimensioni features per ogni scala (adatta al tuo encoder specifico)
    args.low_level_dim = 512
    args.mid_level_dim = 1024
    args.high_level_dim = 1536
    
    return args