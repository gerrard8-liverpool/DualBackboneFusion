import os
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def load_clip_to_cpu_by_name(backbone_name: str):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model


def load_clip_to_cpu(cfg):
    return load_clip_to_cpu_by_name(cfg.MODEL.BACKBONE.NAME)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError(f"Unknown class token position: {self.class_token_position}")

        return prompts


class ResidualFeatureFusion(nn.Module):
    """ViT-anchored feature residual fusion.

    The final linear layer is zero-initialized, so the module is identity at
    initialization: output == normalized ViT feature. This avoids destroying the
    original CLIP image-text alignment before learning.
    """

    def __init__(self, dim=512, hidden_dim=512, dropout=0.0, alpha_init=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, feat_vit, feat_rn):
        feat_vit = F.normalize(feat_vit.float(), dim=-1)
        feat_rn = F.normalize(feat_rn.float(), dim=-1)
        delta = self.net(torch.cat([feat_vit, feat_rn], dim=-1))
        fused = feat_vit + self.alpha * delta
        return F.normalize(fused, dim=-1)


class ResidualTextAdapter(nn.Module):
    def __init__(self, dim=512, hidden_dim=512, dropout=0.0, beta_init=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, feat):
        feat = F.normalize(feat.float(), dim=-1)
        delta = self.net(feat)
        return F.normalize(feat + self.beta * delta, dim=-1)


class TrainableDualEncModules(nn.Module):
    def __init__(self, prompt_learner, fusion_adapter, text_adapter=None):
        super().__init__()
        self.prompt_learner = prompt_learner
        self.fusion_adapter = fusion_adapter
        self.text_adapter = text_adapter if text_adapter is not None else nn.Identity()


class CustomCLIPDualEnc(nn.Module):
    def __init__(self, cfg, classnames, clip_vit, clip_rn):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_vit)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.image_encoder_vit = clip_vit.visual
        self.image_encoder_rn = clip_rn.visual
        self.text_encoder = TextEncoder(clip_vit)
        self.logit_scale = clip_vit.logit_scale
        self.dtype_vit = clip_vit.dtype
        self.dtype_rn = clip_rn.dtype

        dim = int(os.environ.get("DUALENC_DIM", "512"))
        hidden_dim = _env_int("DUALENC_HIDDEN_DIM", 512)
        dropout = _env_float("DUALENC_DROPOUT", 0.0)
        alpha_init = _env_float("DUALENC_ALPHA_INIT", 1.0)
        self.fusion_adapter = ResidualFeatureFusion(dim=dim, hidden_dim=hidden_dim, dropout=dropout, alpha_init=alpha_init)

        self.use_text_adapter = _env_flag("DUALENC_USE_TEXT_ADAPTER", False)
        if self.use_text_adapter:
            beta_init = _env_float("DUALENC_BETA_INIT", 1.0)
            self.text_adapter = ResidualTextAdapter(dim=dim, hidden_dim=hidden_dim, dropout=dropout, beta_init=beta_init)
        else:
            self.text_adapter = None

    def trainable_modules(self):
        return TrainableDualEncModules(self.prompt_learner, self.fusion_adapter, self.text_adapter)

    def encode_text(self):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_batch_size = int(os.environ.get("TEXT_BATCH_SIZE", "0"))
        if text_batch_size > 0 and prompts.shape[0] > text_batch_size:
            text_features_list = []
            for start in range(0, prompts.shape[0], text_batch_size):
                end = start + text_batch_size
                text_features_list.append(self.text_encoder(prompts[start:end], tokenized_prompts[start:end]))
            text_features = torch.cat(text_features_list, dim=0)
        else:
            text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = F.normalize(text_features.float(), dim=-1)
        if self.text_adapter is not None:
            text_features = self.text_adapter(text_features)
        return text_features

    def forward(self, image):
        feat_vit = self.image_encoder_vit(image.type(self.dtype_vit))
        feat_rn = self.image_encoder_rn(image.type(self.dtype_rn))
        image_features = self.fusion_adapter(feat_vit, feat_rn)
        text_features = self.encode_text()
        logit_scale = self.logit_scale.exp().float()
        logits = logit_scale * image_features @ text_features.t()
        return logits


@TRAINER_REGISTRY.register()
class CoOpDualEnc(TrainerX):
    """ViT-anchored dual-backbone feature fusion for CoOp.

    Anchor backbone: cfg.MODEL.BACKBONE.NAME, expected to be ViT-B/16.
    Auxiliary backbone: environment variable DUALENC_AUX_BACKBONE, default RN101.
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
        if cfg.TRAINER.COOP.CSC:
            print("[WARN] CSC=True is not recommended for cross-dataset evaluation because context shape depends on class count.")

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        aux_backbone = os.environ.get("DUALENC_AUX_BACKBONE", "RN101")

        print(f"Loading anchor CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_vit = load_clip_to_cpu(cfg)
        print(f"Loading auxiliary CLIP (backbone: {aux_backbone})")
        clip_rn = load_clip_to_cpu_by_name(aux_backbone)

        if cfg.TRAINER.COOP.PREC in ["fp32", "amp"]:
            clip_vit.float()
            clip_rn.float()

        print("Building CoOpDualEnc custom CLIP")
        self.model = CustomCLIPDualEnc(cfg, classnames, clip_vit, clip_rn)

        print("Freezing CLIP image/text encoders; training prompt learner and fusion adapter")
        for _, param in self.model.named_parameters():
            param.requires_grad_(False)
        for name, param in self.model.prompt_learner.named_parameters():
            param.requires_grad_(True)
        for name, param in self.model.fusion_adapter.named_parameters():
            param.requires_grad_(True)
        if self.model.text_adapter is not None:
            for name, param in self.model.text_adapter.named_parameters():
                param.requires_grad_(True)

        self.model.to(self.device)
        self.model.image_encoder_vit.eval()
        self.model.image_encoder_rn.eval()
        self.model.text_encoder.eval()

        self.trainable = self.model.trainable_modules().to(self.device)
        self.optim = build_optimizer(self.trainable, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("dualenc", self.trainable, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        n_trainable = sum(p.numel() for p in self.trainable.parameters() if p.requires_grad)
        print(f"# trainable params: {n_trainable:,}")
        print(f"[DualEnc] use_text_adapter={self.model.use_text_adapter}")
        print(f"[DualEnc] fusion alpha init={self.model.fusion_adapter.alpha.item():.4f}")

    def forward_backward(self, batch):
        image, label, _ = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC

        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {"loss": loss.item(), "acc": compute_accuracy(output, label)[0].item()}
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        domain = batch.get("domain", torch.zeros_like(label)).to(self.device)
        return image, label, domain

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        model_file = "model-best.pth.tar" if epoch is None else "model.pth.tar-" + str(epoch)
        for name in self.get_model_names():
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f"No model at {model_path}")
            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch_loaded = checkpoint.get("epoch", "unknown")
            val_result = checkpoint.get("val_result", None)

            # Token buffers depend on target class names. They must be rebuilt
            # for the current dataset rather than loaded from the source model.
            for key in [
                "prompt_learner.token_prefix",
                "prompt_learner.token_suffix",
            ]:
                if key in state_dict:
                    state_dict.pop(key)

            incompatible = self._models[name].load_state_dict(state_dict, strict=False)
            print(f"Load {model_path} to {name} (epoch={epoch_loaded}, val_result={val_result})")
            if incompatible.missing_keys:
                print(f"[load_model] missing keys: {incompatible.missing_keys}")
            if incompatible.unexpected_keys:
                print(f"[load_model] unexpected keys: {incompatible.unexpected_keys}")
