import os
import torch
from model import S_Mamba_Patch, \
    Transformer, Informer, Reformer, Flowformer, Flashformer, \
    iTransformer, iInformer, iReformer, iFlowformer, iFlashformer, S_Mamba, \
    Flashformer_M, Flowformer_M, Autoformer, Autoformer_M, Transformer_M, \
    Informer_M, Reformer_M


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_aliases = {
            'S_Mamba_Patch': 'S_Mamba_Patch_FrontAFFB',
            'S_Mamba_AFFB': 'S_Mamba_FrontAFFB',
            'S_Mamba_GateAFFB': 'S_Mamba_FrontGateAFFB',
            'S_Mamba_BiIDMB': 'S_Mamba_BiIDMB',
            'S_Mamba_BiIDMB_GateAFFB': 'S_Mamba_BiIDMB_GateAFFB',
            'S_Mamba_Stationary_BiIDMB_GateAFFB': 'S_Mamba_Stationary_BiIDMB_GateAFFB',
            'S_Mamba_Stationary_BiIDMB_GateAFFB_V1.0': 'S_Mamba_Stationary_BiIDMB_GateAFFB_V1_0',
            'S_Mamba_Stationary_BiIDMB_GateAFFB_V1.1': 'S_Mamba_Stationary_BiIDMB_GateAFFB_V1_1',
            'S_Mamba_Stationary_BiIDMB_GateAFFB_V1.2': 'S_Mamba_Stationary_BiIDMB_GateAFFB_V1_2',
            'S_Mamba_Stationary_BiIDMB_GateAFFB_V1.3': 'S_Mamba_Stationary_BiIDMB_GateAFFB_V1_3',
        }

        self.model_dict = {
            'S_Mamba': S_Mamba,
            'S_Mamba_Patch_FrontAFFB': S_Mamba_Patch,
            'S_Mamba_FrontAFFB': S_Mamba,
            'S_Mamba_FrontGateAFFB': S_Mamba,
            'S_Mamba_BiIDMB': S_Mamba,
            'S_Mamba_BiIDMB_GateAFFB': S_Mamba,
            'S_Mamba_Stationary_BiIDMB_GateAFFB': S_Mamba,
            'S_Mamba_Stationary_BiIDMB_GateAFFB_V1_0': S_Mamba,
            'S_Mamba_Stationary_BiIDMB_GateAFFB_V1_1': S_Mamba,
            'S_Mamba_Stationary_BiIDMB_GateAFFB_V1_2': S_Mamba,
            'S_Mamba_Stationary_BiIDMB_GateAFFB_V1_3': S_Mamba,
            'iTransformer': iTransformer,
            'iInformer': iInformer,
            'iReformer': iReformer,
            'iFlowformer': iFlowformer,
            'iFlashformer': iFlashformer,
            'Transformer': Transformer,
            'Transformer_M': Transformer_M,
            'Informer': Informer,
            'Informer_M': Informer_M,
            'Reformer': Reformer,
            'Reformer_M': Reformer_M,
            'Flowformer': Flowformer,
            'Flowformer_M': Flowformer_M,
            'Flashformer': Flashformer,
            'Flashformer_M': Flashformer_M,
            'Autoformer': Autoformer,
            'Autoformer_M': Autoformer_M,
        }

        self.args.model = self.model_aliases.get(self.args.model, self.args.model)
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
