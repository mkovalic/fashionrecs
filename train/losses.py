import torch
import torch.nn.functional as F


def triplet_loss(a: torch.Tensor, p: torch.Tensor, n: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """Margin-based triplet loss using cosine distance."""
    pos = 1 - (a * p).sum(dim=-1)
    neg = 1 - (a * n).sum(dim=-1)
    return F.relu(pos - neg + margin).mean()


class InfoNCELoss(torch.nn.Module):
    """Symmetric InfoNCE/NT-Xent loss over a batch of positive pairs."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        logits = (a @ b.T) / self.temperature
        labels = torch.arange(a.size(0), device=a.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_ab + loss_ba)


__all__ = ["triplet_loss", "InfoNCELoss"]
