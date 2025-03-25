import torch
from torch import nn
import torch.nn.functional as F
import timm
import config as CFG
from kan import KAN
from mamba_ssm import Mamba

from utils import pre_model_cfg


def my_mamba(dim=128, d_state=16, d_conv=4, expand=2):
    # input: (N, L, C)
    mamba = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim,  # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,  # Local convolution width
        expand=2,  # Block expansion factor
    )
    # output: (N, L, C)
    return mamba


class ImageEncoder(nn.Module):
    def __init__(self, model_name=CFG.model_name, trainable=CFG.trainable):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained=False, num_classes=0, global_pool="avg"
        )
        local_weight_path = pre_model_cfg[model_name]
        local_weight = torch.load(local_weight_path)
        self.model.load_state_dict(local_weight, strict=False)
        print(self.model.default_cfg['input_size'])
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class SpotEncoder(nn.Module):
    def __init__(self, emb=CFG.spot_embedding):
        super().__init__()

    def forward(self, x):

        return x


class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=CFG.projection_dim,
            dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        # self.fc = nn.Linear(projection_dim, projection_dim)

        C_in = 64
        self.fc_in = nn.Linear(projection_dim, C_in)
        self.kan = KAN(width=[C_in, C_in])
        # self.tanh = nn.Tanh()
        self.fc_out = nn.Linear(C_in, projection_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        # x = self.fc(x)

        x = self.fc_in(x)
        x = self.kan(x)
        # x = self.gelu(x)
        x = self.fc_out(x)

        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        spot_embedding=CFG.spot_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.spot_encoder = SpotEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)  # 2048 for resnet50
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding)  # 3467 shared hvgs
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        # [256, 224, 3, 224] -> [256, 3, 224, 224]
        batch_image = batch["image"].permute(0, 2, 1, 3)
        # input:[256, 3, 224, 224] -> output:[256, 2048]
        image_features = self.image_encoder(batch_image)
        # [256, 3467]
        batch_spot = batch["reduced_expression"]
        # input:[256, 3467] -> output:[256, 3467]
        spot_features = self.spot_encoder(batch_spot)  # ours
        # spot_features = batch_spot  # BLEEP

        # Getting Image and Spot Embeddings (with same dimension)
        # input:[256, 2048] -> output:[256, 256]
        image_embeddings = self.image_projection(image_features)
        # input:[256, 3467] -> output:[256, 256]
        spot_embeddings = self.spot_projection(spot_features)

        # Calculating the Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax(((images_similarity + spots_similarity) / 2) / self.temperature, dim=-1)
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + spots_loss) / 2.0  # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


if __name__ == '__main__':
    images = torch.randn(8, 224, 3, 224).cuda()
    reduced_expression = torch.randn(8, 3467).cuda()
    batch = {
        'image': images,
        'reduced_expression': reduced_expression
    }

    CLIP = CLIPModel().cuda()
    loss = CLIP(batch)
    print("debug")

