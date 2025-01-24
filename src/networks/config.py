from src.networks.decoders import Decoders

def get_model(cfg,bound):
    c_dim = cfg['model']['c_dim']  # feature dimensions
    truncation = cfg['model']['truncation']
    learnable_beta = cfg['rendering']['learnable_beta']

    decoder = Decoders(c_dim=c_dim, truncation=truncation, learnable_beta=learnable_beta,bound=bound)

    return decoder
