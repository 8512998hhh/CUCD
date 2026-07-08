import abc

import torch
from torch.utils import data as torch_data
from torch import Tensor
from tqdm import tqdm

from utils import datasets_points, helpers
import wandb

EPS = 10e-05


class AbstractMeasurer(abc.ABC):
    def __init__(self, threshold: float = 0.5, name: str = None):

        self.threshold = threshold
        self.name = name

        # urban mapping
        self.TP_seg_cont = self.TN_seg_cont = self.FP_seg_cont = self.FN_seg_cont = 0
        self.TP_seg_fl = self.TN_seg_fl = self.FP_seg_fl = self.FN_seg_fl = 0

        # urban change | cch -> continuous change | flch -> first to last change
        self.TP_ch_cont = self.TN_ch_cont = self.FP_ch_cont = self.FN_ch_cont = 0
        self.TP_ch_fl = self.TN_ch_fl = self.FP_ch_fl = self.FN_ch_fl = 0

    def add_sample(self, *args, **kwargs):
        raise NotImplementedError("add_sample method must be implemented in the subclass.")

    def _update_metrics(self, y: Tensor, y_hat: Tensor, attr_name: str, mask: Tensor = None):
        y = y.bool()
        y_hat = y_hat > self.threshold

        tp_attr = f'TP_{attr_name}'
        tn_attr = f'TN_{attr_name}'
        fp_attr = f'FP_{attr_name}'
        fn_attr = f'FN_{attr_name}'

        tp = (y & y_hat).float()
        tn = (~y & ~y_hat).float()
        fp = (y_hat & ~y).float()
        fn = (~y_hat & y).float()

        if mask is not None:
            tp[mask] = float('nan')
            tn[mask] = float('nan')
            fp[mask] = float('nan')
            fn[mask] = float('nan')

        setattr(self, tp_attr, getattr(self, tp_attr) + torch.nansum(tp).float().item())
        setattr(self, tn_attr, getattr(self, tn_attr) + torch.nansum(tn).float().item())
        setattr(self, fp_attr, getattr(self, fp_attr) + torch.nansum(fp).float().item())
        setattr(self, fn_attr, getattr(self, fn_attr) + torch.nansum(fn).float().item())


class MultiTaskMeasurer(AbstractMeasurer):
    def __init__(self, threshold: float = 0.5, name: str = None):
        super().__init__(threshold, name)

    def add_sample(self, y_seg: Tensor, y_hat_seg: Tensor, y_ch: Tensor, y_hat_ch: Tensor, mask: Tensor = None):

        # urban mapping
        if y_seg is not None:
            self._update_metrics(y_seg, y_hat_seg, 'seg_cont', mask)
            self._update_metrics(y_seg[:, [0, -1]], y_hat_seg[:, [0, -1]], 'seg_fl', mask)

        # urban change
        if y_hat_ch.size(1) > 1:
            self._update_metrics(y_ch[:, :-1], y_hat_ch[:, :-1], 'ch_cont', mask)
        self._update_metrics(y_ch[:, -1], y_hat_ch[:, -1], 'ch_fl', mask)



def run_quantitative_evaluation(net, cfg, device, run_type: str, enable_mti: bool = False,
                                mti_edge_setting: str = 'dense') -> MultiTaskMeasurer:
    tile_size = cfg.AUGMENTATION.EVAL_PATCH_SIZE
    ds = datasets_points.create_eval_dataset(cfg, run_type, tiling=tile_size)

    net.to(device)
    net.eval()

    m = MultiTaskMeasurer()

    batch_size = 1 if enable_mti else cfg.TRAINER.BATCH_SIZE
    dataloader = torch_data.DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False)


    for step, item in enumerate(tqdm(dataloader, desc=f"Evaluating {run_type}", unit="batch")):
        x, y_seg = item['x'].to(device), item['y']
        timestamps = item['timestamps'].to(device)
        #print(timestamps)
        
        current_T = x.size(1)
        if current_T < cfg.DATALOADER.TIMESERIES_LENGTH:
            print(f"Current time series length {current_T} is less than {cfg.DATALOADER.TIMESERIES_LENGTH}")
            continue
            
        current_edges_cyclic = helpers.get_edges(current_T, 'cyclic')
        current_edges_mti = helpers.get_edges(current_T, mti_edge_setting)
        y_ch = helpers.get_ch(y_seg, current_edges_cyclic)
        
        with torch.no_grad():
            if enable_mti:
                if hasattr(net, 'module'):
                    o_seg = net.module.inference(x, current_edges_mti, timestamps)
                else:
                    o_seg = net.inference(x, current_edges_mti, timestamps)
                o_ch = helpers.get_ch(o_seg, current_edges_cyclic)
            else:
                logits_ch, logits_seg = net(x, current_edges_cyclic, timestamps)
                o_ch, o_seg = torch.sigmoid(logits_ch).detach(), torch.sigmoid(logits_seg).detach()
                
        m.add_sample(y_seg.cpu(), o_seg.cpu(), y_ch.cpu(), o_ch.cpu())

    return m


def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int) -> float:
    # 这个函数本身不需要修改，依然完美运转
    m = run_quantitative_evaluation(net, cfg, device, run_type)

    f1_seg_cont = f1_score(m.TP_seg_cont, m.FP_seg_cont, m.FN_seg_cont)
    f1_seg_fl = f1_score(m.TP_seg_fl, m.FP_seg_fl, m.FN_seg_fl)
    f1_ch_cont = f1_score(m.TP_ch_cont, m.FP_ch_cont, m.FN_ch_cont)
    f1_ch_fl = f1_score(m.TP_ch_fl, m.FP_ch_fl, m.FN_ch_fl)
    f1 = (f1_seg_cont + f1_seg_fl + f1_ch_cont + f1_ch_fl) / 4

    wandb.log({
        f'{run_type} f1': f1,
        f'{run_type} f1 seg cont': f1_seg_cont,
        f'{run_type} f1 seg fl': f1_seg_fl,
        f'{run_type} f1 ch cont': f1_ch_cont,
        f'{run_type} f1 ch fl': f1_ch_fl,
        'step': step, 'epoch': epoch,
    })

    return f1


def precision(tp: int, fp: int) -> float:
    return tp / (tp + fp + EPS)


def recall(tp: int, fn: int) -> float:
    return tp / (tp + fn + EPS)


def f1_score(tp: int, fp: int, fn: int) -> float:
    p = precision(tp, fp)
    r = recall(tp, fn)
    return (2 * p * r) / (p + r + EPS)


def iou(tp: int, fp: int, fn: int) -> float:
    return tp / (tp + fp + fn + EPS)
