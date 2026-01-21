import torch
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_, lecun_normal_
import math
from functools import partial
import torch.nn.functional as F
from src.util.EnhancedTransformerEncoder import EnhancedTransformerEncoderLayer


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class MetaGE(nn.Module):
    def __init__(self,
                 embed_dim=128, depth=4, 
                 decoder_embed_dim=128,
                 num_classes=1000,
                 num_kmers = 64,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 num_attention_head=8,
                 is_pretrain=False,
                 device=None, dtype=None,
                 **kwargs):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs) 
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim
        self.is_pretrain = is_pretrain
        self.position_sequence_embeddinng = nn.Embedding(num_kmers+1, embed_dim)
        self.centrility_embeddinng = nn.Embedding(num_kmers+1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_cls_token = 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_kmers + num_cls_token, embed_dim))
        self.d1 = nn.Linear(embed_dim, 64)
        self.d2 = nn.Linear(64, num_classes)
        self.enhancedtransformer_encoder_1_ = EnhancedTransformerEncoderLayer(embedding_dim = embed_dim, ffn_embedding_dim = int(embed_dim)*4, num_attention_heads = num_attention_head)
        self.enhancedtransformer_encoder_2_ = EnhancedTransformerEncoderLayer(embedding_dim = embed_dim, ffn_embedding_dim = int(embed_dim)*4, num_attention_heads = num_attention_head)
        self.enhancedtransformer_encoder_3_ = EnhancedTransformerEncoderLayer(embedding_dim = embed_dim, ffn_embedding_dim = int(embed_dim)*4, num_attention_heads = num_attention_head)
        self.enhancedtransformer_encoder_4_ = EnhancedTransformerEncoderLayer(embedding_dim = embed_dim, ffn_embedding_dim = int(embed_dim)*4, num_attention_heads = num_attention_head)
        self.enhancedtransformer_encoder_5_ = EnhancedTransformerEncoderLayer(embedding_dim = embed_dim, ffn_embedding_dim = int(embed_dim)*4, num_attention_heads = num_attention_head)
        self.enhancedtransformer_encoder_6_ = EnhancedTransformerEncoderLayer(embedding_dim = embed_dim, ffn_embedding_dim = int(embed_dim)*4, num_attention_heads = num_attention_head)
        self.enhancedtransformer_encoder_7_ = EnhancedTransformerEncoderLayer(embedding_dim = embed_dim, ffn_embedding_dim = int(embed_dim)*4, num_attention_heads = num_attention_head)
        self.enhancedtransformer_encoder_8_ = EnhancedTransformerEncoderLayer(embedding_dim = embed_dim, ffn_embedding_dim = int(embed_dim)*4, num_attention_heads = num_attention_head)
        self.num_kmers = num_kmers
        num_cls_token = 1
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.pos_drop = nn.Dropout(p=drop_rate)

        if is_pretrain:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.edge_weight = nn.Parameter(torch.ones(1, 8, 65, 65))
            self.dense=nn.Linear(256, 256)
            self.head1 = nn.Linear(self.num_features, int(self.num_features/2))
            self.leaky_relu = nn.LeakyReLU()
            self.dropout = nn.Dropout(0.3)
            self.batchnorm1 = nn.BatchNorm1d(int(self.num_features/2))
            self.head2 = nn.Linear(int(self.num_features/2), num_classes)
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.pad_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if not self.is_pretrain:
            self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pad_tokens, std=.02)
        if self.is_pretrain:
            trunc_normal_(self.decoder_embed.weight, std=.02)
            trunc_normal_(self.mask_token, std=.02)
        self.apply(partial(_init_weights, n_layer=depth,))
        
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "pad_tokens", "position_sequence_embeddinng.weight", "centrility_embeddinng.weight"}

    def random_masking(self, x, mask_ratio, pad0):
        B, N, F = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        mask = torch.ones([B, N], device=x.device)
        if pad0 is not None:
            mask = mask * (~pad0)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return ids_keep, mask, ids_restore

    def forward_encoder(self, x, edge, mask_ratio, if_mask=True,):
        x1 = x[:,:self.num_kmers]
        pad0 = (x1 == 64)
        pad = torch.cat([pad0, torch.zeros(pad0.size(0), 1, dtype=torch.bool, device=pad0.device)], dim=1)
        x2 = x[:,self.num_kmers:]
        x1 = self.position_sequence_embeddinng(x1)
        x2 = self.centrility_embeddinng(x2)
        x = x1 + x2
        pad_tokens = self.pad_tokens
        x = torch.where(pad0.unsqueeze(-1) == 1, pad_tokens, x)
        x = x + self.pos_embed[:, :-1, :]
        if if_mask:
            ids_keep, mask, ids_restore = self.random_masking(x1, mask_ratio, pad0)
            mask_tokens = self.mask_token
            x = torch.where(mask.unsqueeze(-1) == 1, mask_tokens, x)
        cls_token = self.cls_token + self.pos_embed[:, -1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)
        x = self.pos_drop(x)
        if if_mask:
            B, N, D = x.shape
            x = x.transpose(0, 1)
            x, _ = self.enhancedtransformer_encoder_1_(x, edge, self_attn_padding_mask=pad)
            x, _ = self.enhancedtransformer_encoder_2_(x, edge, self_attn_padding_mask=pad)
            x, _ = self.enhancedtransformer_encoder_3_(x, edge, self_attn_padding_mask=pad)
            x, _ = self.enhancedtransformer_encoder_4_(x, edge, self_attn_padding_mask=pad)
            x, _ = self.enhancedtransformer_encoder_5_(x, edge, self_attn_padding_mask = pad)
            x, _ = self.enhancedtransformer_encoder_6_(x, edge, self_attn_padding_mask = pad)
            x, _ = self.enhancedtransformer_encoder_7_(x, edge, self_attn_padding_mask = pad)
            x, _ = self.enhancedtransformer_encoder_8_(x, edge, self_attn_padding_mask = pad)
            x = x.transpose(0, 1)
            x_raw = x
            x = self.decoder_embed(x)
        else:
            B, N, D = x.shape
            edge[:, :, N-1, :] = 1
            edge[:, :, :, N-1] = 1
            x = x.transpose(0, 1)

            x, _ = self.enhancedtransformer_encoder_1_(x,edge, self_attn_padding_mask = pad)
            x, _ = self.enhancedtransformer_encoder_2_(x,edge, self_attn_padding_mask = pad)
            x, _ = self.enhancedtransformer_encoder_3_(x,edge, self_attn_padding_mask = pad)
            x, _ = self.enhancedtransformer_encoder_4_(x,edge, self_attn_padding_mask = pad)
            x, _ = self.enhancedtransformer_encoder_5_(x,edge, self_attn_padding_mask = pad)
            x, _ = self.enhancedtransformer_encoder_6_(x,edge, self_attn_padding_mask = pad)
            x, _ = self.enhancedtransformer_encoder_7_(x,edge, self_attn_padding_mask = pad)
            x, _ = self.enhancedtransformer_encoder_8_(x,edge, self_attn_padding_mask = pad)
            x = x.transpose(0, 1)
        if if_mask:
            return x[:, :-1, :], mask, ids_restore, x_raw
        else:
            return x, pad

    def forward_rec_loss(self, target, pred, mask):
        criterion = torch.nn.CrossEntropyLoss()
        vocab_size = pred.shape[2]
        pred_reshaped = pred.reshape(-1, vocab_size)
        target_reshaped = target.reshape(-1)
        mask_reshaped = mask.reshape(-1)
        valid_indices = mask_reshaped.bool()
        valid_pred = pred_reshaped[valid_indices]
        valid_target = target_reshaped[valid_indices].long()
        loss = criterion(valid_pred, valid_target)
        return loss

    def forward(self, sequences, edge, mask_ratio=0.9,):
        if self.is_pretrain:
            raw_sequences = sequences.clone()
            latent, mask, ids_restore, x_raw = self.forward_encoder(sequences, edge,
                    mask_ratio=mask_ratio,)
            pred = latent
            target_kmer_sequence = raw_sequences[:,:self.num_kmers]
            loss = self.forward_rec_loss(target_kmer_sequence, pred, mask)
            return loss, pred, mask, x_raw
        else:
            x, pad = self.forward_encoder(sequences, edge, mask_ratio=mask_ratio, if_mask=False)
            # cls_token = x[:, -1, :]
            x = self.head(x[:, -1, :])
            return x
        

def MetaGE_pretrain(**kwargs):
    model = MetaGE(
        is_pretrain=True, **kwargs)
    return model

def MetaGE_classifier(**kwargs):
    model = MetaGE(
        is_pretrain=False,**kwargs)
    return model