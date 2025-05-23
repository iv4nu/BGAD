from torch import nn
# FrEIA (https://github.com/VLL-HD/FrEIA/)
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import matplotlib.pyplot as plt
import os

'''
def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2*dims_in), nn.ReLU(), nn.Linear(2*dims_in, dims_out))
'''

def flow_model(args, in_channels):
    coder = Ff.SequenceINN(in_channels)
    print('Normalizing Flow => Feature Dimension: ', in_channels)
    for k in range(args.coupling_layers):
        coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=args.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder

'''
def conditional_flow_model(args, in_channels):
    coder = Ff.SequenceINN(in_channels)
    print('Conditional Normalizing Flow => Feature Dimension: ', in_channels)
    for k in range(args.coupling_layers):  # 8
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(args.pos_embed_dim,), subnet_constructor=subnet_fc, affine_clamping=args.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder
    '''
 #in caso non va bene cancella tutto ciò che va da qui    
def save_activation_histogram(tensor, epoch, layer_id, save_dir='/kaggle/working/activation_histograms'):
    os.makedirs(save_dir, exist_ok=True)
    values = tensor.detach().cpu().numpy().flatten()
    plt.figure()
    plt.hist(values, bins=100, color='skyblue')
    plt.title(f"Activation Histogram – Epoch {epoch} – Layer {layer_id}")
    plt.xlabel("Activation")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"epoch{epoch}_layer{layer_id}.png"))
    plt.close()

def subnet_fc(dims_in, dims_out, layer_id=None, args=None):
    layers = nn.Sequential(
        nn.Linear(dims_in, 2 * dims_in),
        nn.ReLU(),  # <-- attivazione da tracciare
        nn.Linear(2 * dims_in, dims_out)
    )

    if args is not None and layer_id is not None:
        def hook(module, input, output):
            if hasattr(args, 'current_epoch'):
                save_activation_histogram(output, args.current_epoch, layer_id)
        layers[1].register_forward_hook(hook)  # 👈 hook sulla ReLU

    return layers
def conditional_flow_model(args, in_channels):
    print('ok')
    coder = Ff.SequenceINN(in_channels)
    print('Conditional Normalizing Flow => Feature Dimension: ', in_channels)
    for k in range(args.coupling_layers):  # 8
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(args.pos_embed_dim,), subnet_constructor=lambda in_dim, out_dim: subnet_fc(in_dim, out_dim, layer_id=k, args=args),
affine_clamping=args.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder
    # a qui
''' 
def conditional_flow_model(args, in_channels):
    return build_optimized_flow_model(args,in_channels)

def build_optimized_flow_model(args,input_dim):
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
            permute_soft=True
        )
    return flow'''

