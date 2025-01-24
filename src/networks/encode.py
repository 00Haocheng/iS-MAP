# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. and https://github.com/HengyiWang/Co-SLAM/tree/main/model/encodings

import torch
import torch.nn as nn
import tinycudann as tcnn
import numpy as np
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)#创建从0到max_freq的有N_freqs个值的序列[1,2,4,8,16,32]
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))#这里就是编码sin/cos（2**n*pai*p）
                out_dim += d#三维位姿不断变为更高纬度

        self.embed_fns = embed_fns#位姿编码函数,一个list，里面是2的0-n次方的编码
        self.out_dim = out_dim#位姿编码后的维度

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim  #返回编码函数及编码后的维度





def get_hash_encoder(encoding, input_dim=3,
                degree=4, n_bins=16, n_frequencies=12,
                n_levels=18, level_dim=2,
                base_resolution=16, log2_hashmap_size=19,
                desired_resolution=512):
    # Dense grid encoding
    if 'dense' in encoding.lower():
        n_levels = 4
        per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (n_levels - 1))
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "Grid",
                "type": "Dense",
                "n_levels": n_levels,
                "n_features_per_level": level_dim,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
                "interpolation": "Linear"},
            dtype=torch.float
        )
        out_dim = embed.n_output_dims

    # Sparse grid encoding
    # elif 'hash' in encoding.lower() or 'tiled' in encoding.lower():
    #     log2_hashmap_size = 14,
    #     desired_resolution = 400
    #     print('Hash size', log2_hashmap_size)
    #     per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (n_levels - 1))
    #     embed = tcnn.Encoding(
    #         n_input_dims=input_dim,
    #         encoding_config={
    #             "otype": 'HashGrid',
    #             "n_levels": n_levels,
    #             "n_features_per_level": level_dim,
    #             "log2_hashmap_size": log2_hashmap_size,
    #             "base_resolution": base_resolution,
    #             "per_level_scale": per_level_scale
    #         },
    #         dtype=torch.float
    #     )
    #     out_dim = embed.n_output_dims
    elif 'hash' in encoding.lower() or 'tiled' in encoding.lower():
        print('Hash size', log2_hashmap_size)
        per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (n_levels - 1))
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": 'HashGrid',
                "n_levels": n_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale
            },
            dtype=torch.float
        )
        out_dim = embed.n_output_dims

    # Spherical harmonics encoding
    elif 'spherical' in encoding.lower():
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": degree,
            },
            dtype=torch.float
        )
        out_dim = embed.n_output_dims

    # OneBlob encoding
    elif 'blob' in encoding.lower():
        print('Use blob')
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "OneBlob",  # Component type.
                "n_bins": n_bins
            },
            dtype=torch.float
        )
        out_dim = embed.n_output_dims

    # Frequency encoding
    elif 'freq' in encoding.lower():
        print('Use frequency')
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": n_frequencies
            },
            dtype=torch.float
        )
        out_dim = embed.n_output_dims

    # Identity encoding
    elif 'identity' in encoding.lower():
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "Identity"
            },
            dtype=torch.float
        )
        out_dim = embed.n_output_dims

    return embed, out_dim