import argparse
import torch

#if __name__ == '__main__':
class Config:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='SSLPDL')
        self._add_arguments()

    def _add_arguments(self):
        # model define
        self.parser.add_argument('--core_op', type=str, default='DCNv3_pytorch', help='core operation of InternImage')
        self.parser.add_argument('--activation', type=str, default='GELU', help='activation type')
        self.parser.add_argument('--normalization', type=str, default='LN', help='normalization type') 
        self.parser.add_argument('--drop_path_type', type=str, default='linear', help='')
        self.parser.add_argument('--embed_dim', type=int, default=1024, help='dimension of embedding')
        self.parser.add_argument('--channels', type=int, default=64, help='dimension of model')
        self.parser.add_argument('--dropout', type=float, default=0., help='dropout') 
        self.parser.add_argument('--drop_path_rate', type=float, default=0.1, help='drop path rate')
        self.parser.add_argument('--mlp_ratio', type=float, default=4., help='ratio of mlp hidden features to input channels')
        self.parser.add_argument('--offset_scale', type=float, default=1.0, help='offset scale')
        self.parser.add_argument('--cls_scale', type=float, default=1.5, help='Whether to use class scale.')
        self.parser.add_argument('--post_norm', type=bool, default=False, help='whether to use post normalization')
        self.parser.add_argument('--layer_scale', type=bool, default=False, help='whether to scale layer')

        # SSLPDL
        self.parser.add_argument('--num_classes', type=int, default=16, help='number of classes')
        self.parser.add_argument('--in_channels', type=int, default=16, help='num of variables')
        self.parser.add_argument('--img_size', type=list, default=(224, 128), help='input size')
        self.parser.add_argument('--patch_size', type=list, default=(4, 16, 16), help='patch size')


