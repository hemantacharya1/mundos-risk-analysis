import torch

def test_mean_pooling_mask():
    emb = torch.arange(2*4*3, dtype=torch.float32).reshape(2,4,3)
    mask = torch.tensor([[1,1,1,0],[1,0,0,0]]).unsqueeze(-1)
    sums = (emb * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    mean = sums / counts
    manual0 = emb[0,:3].mean(dim=0)
    assert torch.allclose(mean[0], manual0)
    assert torch.allclose(mean[1], emb[1,0])
