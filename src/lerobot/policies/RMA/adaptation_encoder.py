from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class AdaptationEncoderConfig:
    action_dim: int
    plucker_dim: int
    visual_backbone: str = "facebook/dinov2-small"
    visual_feature_dim: int | None = None
    action_feature_dim: int = 128
    hidden_dim: int = 512
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_ff_dim: int = 2048
    dropout: float = 0.1
    freeze_visual: bool = True
    use_cls_token: bool = True
    image_size: int = 224


class FrozenDinoV2Encoder(nn.Module):
    def __init__(self, model_name: str, freeze: bool = True, image_size: int = 224):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for the DINOv2 visual backbone. "
                "Install transformers in the smolvla conda environment."
            ) from exc

        self.model = AutoModel.from_pretrained(model_name)
        self.image_size = image_size
        self.feature_dim = self.model.config.hidden_size

        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(param.requires_grad for param in self.model.parameters()):
            self.model.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        images = images.float()
        if images.max() > 2.0:
            images = images / 255.0

        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = (images - mean) / std

        output = self.model(pixel_values=images)
        return output.last_hidden_state[:, 0]


class AdaptationEncoder(nn.Module):
    def __init__(self, config: AdaptationEncoderConfig):
        super().__init__()
        self.config = config
        self.visual_encoder = FrozenDinoV2Encoder(
            config.visual_backbone,
            freeze=config.freeze_visual,
            image_size=config.image_size,
        )

        visual_feature_dim = config.visual_feature_dim or self.visual_encoder.feature_dim
        self.action_encoder = nn.Sequential(
            nn.Linear(config.action_dim, config.action_feature_dim),
            nn.GELU(),
            nn.LayerNorm(config.action_feature_dim),
            nn.Linear(config.action_feature_dim, config.action_feature_dim),
            nn.GELU(),
        )
        self.fusion_projection = nn.Sequential(
            nn.Linear(visual_feature_dim + config.action_feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
        )

        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        else:
            self.cls_token = None

        self.regression_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.plucker_dim),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, images: Tensor, actions: Tensor) -> Tensor:
        if images.ndim != 5:
            raise ValueError(f"Expected images with shape (B, T, C, H, W), got {tuple(images.shape)}")
        if actions.ndim != 3:
            raise ValueError(f"Expected actions with shape (B, T, A), got {tuple(actions.shape)}")
        if images.shape[:2] != actions.shape[:2]:
            raise ValueError(
                f"Image/action sequence mismatch: images={tuple(images.shape)}, actions={tuple(actions.shape)}"
            )

        batch_size, seq_len = images.shape[:2]
        flat_images = images.reshape(batch_size * seq_len, *images.shape[2:])
        visual_features = self.visual_encoder(flat_images).reshape(batch_size, seq_len, -1)
        action_features = self.action_encoder(actions.float())

        tokens = self.fusion_projection(torch.cat([visual_features, action_features], dim=-1))
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_token, tokens], dim=1)

        tokens = self.temporal_transformer(tokens)
        trajectory_feature = tokens[:, 0] if self.cls_token is not None else tokens.mean(dim=1)
        return self.regression_head(trajectory_feature)
