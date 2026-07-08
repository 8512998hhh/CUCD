import torch
from model.model import ContUrbanCDMambaModel
from configs.sn7_active import cfg

model = ContUrbanCDMambaModel(cfg)
model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED_PATH))
model.eval()

x = torch.randn(1, 5, 3, 768, 1024)
edges = helpers.get_edges(cfg.DATALOADER.TIMESERIES_LENGTH, "adjacent")
logits_ch, logits_seg = model(x, edges)
print(logits_ch.shape, logits_seg.shape)