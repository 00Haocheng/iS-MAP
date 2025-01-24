import os
import torch

class Logger(object):
    """
    Save checkpoints to file.

    """

    def __init__(self, ismap):
        self.verbose = ismap.verbose
        self.ckptsdir = ismap.ckptsdir
        self.gt_c2w_list = ismap.gt_c2w_list
        self.shared_decoders = ismap.shared_decoders
        self.estimate_c2w_list = ismap.estimate_c2w_list

    def log(self, idx, keyframe_list,allplanes=None):
        path = os.path.join(self.ckptsdir, '{:05d}.tar'.format(idx))
        torch.save({
            'decoder_state_dict': self.shared_decoders.state_dict(),
            'gt_c2w_list': self.gt_c2w_list,
            'estimate_c2w_list': self.estimate_c2w_list,
            'keyframe_list': keyframe_list,
            'idx': idx,
            'allplanes': allplanes,
        }, path, _use_new_zipfile_serialization=False)

        # if self.verbose:
        print('Saved checkpoints at', path)
