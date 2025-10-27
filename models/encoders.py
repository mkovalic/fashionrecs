import torch
import torch.nn as nn
import open_clip
from PIL import Image

class FrozenOpenCLIP(nn.Module):
    def __init__(self, backbone="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        super().__init__()
        # Support hf-hub:* identifiers where pretrained is handled by OpenCLIP
        if str(backbone).startswith("hf-hub:"):
            model, _, preprocess = open_clip.create_model_and_transforms(backbone)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
        self.model = model
        self.preprocess = preprocess
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # Infer output dim robustly (works across CLIP/SigLIP variants)
        self._embed_dim = None
        try:
            self._embed_dim = getattr(self.model, "embed_dim", None)
        except Exception:
            self._embed_dim = None
        if self._embed_dim is None:
            # Fallback: run a 1x dummy forward on CPU
            with torch.no_grad():
                dummy = Image.new("RGB", (224, 224))
                t = preprocess(dummy).unsqueeze(0)
                z = self.model.encode_image(t)
                self._embed_dim = z.shape[-1]

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images expected as preprocessed tensors [B,3,H,W]
        # Many OpenCLIP models support normalize=True in encode_image; we apply manual L2 norm for safety.
        feats = self.model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def get_preprocess(self):
        return self.preprocess

    @property
    def embed_dim(self):
        return int(self._embed_dim)
