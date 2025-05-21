from torch import nn
# FrEIA (https://github.com/VLL-HD/FrEIA/)
import FrEIA.framework as Ff
import FrEIA.modules as Fm


def subnet_fc(dims_in, dims_out):
    return nn.Sequential(nn.Linear(dims_in, 2*dims_in), nn.ReLU(), nn.Linear(2*dims_in, dims_out))


def flow_model(args, in_channels):
    coder = Ff.SequenceINN(in_channels)
    print('Normalizing Flow => Feature Dimension: ', in_channels)
    for k in range(args.coupling_layers):
        coder.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, affine_clamping=args.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder


'''def conditional_flow_model(args, in_channels):
    coder = Ff.SequenceINN(in_channels)
    print('Conditional Normalizing Flow => Feature Dimension: ', in_channels)
    for k in range(args.coupling_layers):  # 8
        coder.append(Fm.AllInOneBlock, cond=0, cond_shape=(args.pos_embed_dim,), subnet_constructor=subnet_fc, affine_clamping=args.clamp_alpha,
            global_affine_type='SOFTPLUS', permute_soft=True)
    return coder
    
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
    return flow