import os
import torch
from collections import OrderedDict
import glob

class Saver(object):

    def __init__(self, args):
        self.args = args

    def save_checkpoint(self, state, filename):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.args.savepath, filename)
        torch.save(state, filename)

    def save_checkpoint_v2(self, state, checkpoint_path):
            """Saves checkpoint to disk"""
            # filename = os.path.join(self.args.save_folder, filename)
            torch.save(state, checkpoint_path)
