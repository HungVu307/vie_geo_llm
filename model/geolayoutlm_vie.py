# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import sys
import logging

import torch
from torch import nn
import random

# from bros import BrosConfig, BrosTokenizer
from bros import BrosConfig
from transformers import AutoTokenizer
from bros import GeoLayoutLMModel, PairGeometricHead, MultiPairsGeometricHead
from torch.nn import Embedding
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.init as init

def add_gaussian_blur(image_tensor, kernel_size=3, sigma=1.0):
    blur = transforms.GaussianBlur(kernel_size, sigma=sigma)
    blurred_image_tensor = blur(image_tensor)
    return blurred_image_tensor



logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s %(filename)s:%(lineno)d [%(levelname)s] %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class GeoLayoutLMVIEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.model_cfg = cfg.model
        if self.model_cfg.backbone in [
            "alibaba-damo/geolayoutlm-base-uncased",
            "alibaba-damo/geolayoutlm-large-uncased",
        ]:
            # backbone
            self.backbone_config = BrosConfig.from_json_file(self.model_cfg.config_json)
            # self.tokenizer = BrosTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
            self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", do_lower_case=True)
            self.geolayoutlm_model = GeoLayoutLMModel(self.backbone_config)
            # task head
            if self.model_cfg.use_inner_id:
                # self.bio_classifier = nn.Linear(self.backbone_config.hidden_size*2, self.model_cfg.n_classes)
                self.bio_classifier = nn.Linear(self.backbone_config.hidden_size*2, cfg.num_class)

            else:
                # self.bio_classifier = nn.Linear(self.backbone_config.hidden_size, self.model_cfg.n_classes)
                self.bio_classifier = nn.Linear(self.backbone_config.hidden_size, cfg.num_class)

            self.pair_geometric_head = PairGeometricHead(self.backbone_config)
            self.multi_pairs_geometric_head = MultiPairsGeometricHead(self.backbone_config)
        else:
            raise ValueError(
                f"Not supported model: self.model_cfg.backbone={self.model_cfg.backbone}"
            )

        self.dropout = nn.Dropout(0.1)
        self.loss_func_labeling = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_func_linking = nn.BCEWithLogitsLoss(reduction='none')

        if (getattr(cfg, 'pretrained_model_file', None) is None) and self.model_cfg.backbone in [
            "alibaba-damo/geolayoutlm-base-uncased",
            "alibaba-damo/geolayoutlm-large-uncased",
        ]:
            self._init_weight()
    

    def _init_weight(self):
        model_path = self.model_cfg.model_ckpt
        logger.info("Initializing weights from {}".format(model_path))
        
        # Load the state_dict
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Separate the state_dict into different components
        backbone_state_dict = {}
        pair_geometric_head_dict = {}
        multi_pairs_geometric_head_dict = {}

        for key, value in state_dict.items():
            if key.startswith("geolayoutlm_model."):
                new_key = key.replace("geolayoutlm_model.", "")
                backbone_state_dict[new_key] = value
                continue
            
            if key.startswith("ptm_head.pair_geometric_head."):
                new_key = key.replace("ptm_head.pair_geometric_head.", "")
                if "pair_direct_cls" in new_key:
                    continue
                pair_geometric_head_dict[new_key] = value
                continue

            if key.startswith("ptm_head.multi_pairs_geometric_head."):
                new_key = key.replace("ptm_head.multi_pairs_geometric_head.", "")
                if "pair_direct_cls" in new_key:
                    continue
                multi_pairs_geometric_head_dict[new_key] = value
                continue

        self.geolayoutlm_model = load_model(self.geolayoutlm_model, backbone_state_dict)
        new_embedding_size = 64000  
        existing_embedding_layer = self.geolayoutlm_model.text_encoder.embeddings.word_embeddings

        new_embedding_layer = Embedding(new_embedding_size, existing_embedding_layer.embedding_dim, padding_idx=existing_embedding_layer.padding_idx)
        new_embedding_layer.weight.data[:existing_embedding_layer.weight.size(0), :] = existing_embedding_layer.weight.data
        self.geolayoutlm_model.text_encoder.embeddings.word_embeddings = new_embedding_layer
        
        if hasattr(self.geolayoutlm_model.config, "vocab_size"):
            self.geolayoutlm_model.config.vocab_size = new_embedding_size


        self.pair_geometric_head = load_model(self.pair_geometric_head, pair_geometric_head_dict)
        self.multi_pairs_geometric_head = load_model(self.multi_pairs_geometric_head, multi_pairs_geometric_head_dict)

        return 0


    # def forward(self, batch):
    #     """ batch is a dict with the following keys:
    #     'image', 'input_ids', 'bbox_4p_normalized',
    #     'attention_mask', 'first_token_idxes', 'block_mask',
    #     'bbox', 'line_rank_id', 'line_rank_inner_id'
    #     """

    #     input_ids = batch["input_ids"]
    #     image = batch["image"]
    #     bbox = batch["bbox"]
    #     bbox_4p_normalized = batch["bbox_4p_normalized"]
    #     attention_mask = batch["attention_mask"]
    #     first_token_idxes = batch["first_token_idxes"]
    #     first_token_idxes_mask = batch["block_mask"]
    #     line_rank_id = batch["line_rank_id"]
    #     line_rank_inner_id = batch["line_rank_inner_id"]

    #     if self.model_cfg.backbone in [
    #         "alibaba-damo/geolayoutlm-base-uncased",
    #         "alibaba-damo/geolayoutlm-large-uncased",
    #     ]:
    #         # sequence_output: [batch_size, seq_len, hidden_size]
    #         # blk_vis_features: [batch_size, block_num, hidden_size]
    #         # text_mm_feat: [batch_size, seq_len, hidden_size]
    #         # vis_mm_feat: [batch_size, 1+block_num, hidden_size]
    #         sequence_output, blk_vis_features, text_mm_feat, vis_mm_feat = self.geolayoutlm_model(
    #             input_ids=input_ids,
    #             image=image,
    #             bbox=bbox,
    #             bbox_4p_normalized=bbox_4p_normalized,
    #             attention_mask=attention_mask,
    #             first_token_idxes=first_token_idxes,
    #             first_token_idxes_mask=first_token_idxes_mask,
    #             line_rank_id=line_rank_id,
    #             line_rank_inner_id=line_rank_inner_id,
    #         )
    def forward(self, input_ids, image, bbox, bbox_4p_normalized, attention_mask, first_token_idxes, first_token_idxes_mask, line_rank_id, line_rank_inner_id):
        """ batch is a dict with the following keys:
        'image', 'input_ids', 'bbox_4p_normalized',
        'attention_mask', 'first_token_idxes', 'block_mask',
        'bbox', 'line_rank_id', 'line_rank_inner_id'
        """



        if self.model_cfg.backbone in [
            "alibaba-damo/geolayoutlm-base-uncased",
            "alibaba-damo/geolayoutlm-large-uncased",
        ]:
            # sequence_output: [batch_size, seq_len, hidden_size]
            # blk_vis_features: [batch_size, block_num, hidden_size]
            # text_mm_feat: [batch_size, seq_len, hidden_size]
            # vis_mm_feat: [batch_size, 1+block_num, hidden_size]
            sequence_output, blk_vis_features, text_mm_feat, vis_mm_feat = self.geolayoutlm_model(
                input_ids=input_ids,
                image=image,
                bbox=bbox,
                bbox_4p_normalized=bbox_4p_normalized,
                attention_mask=attention_mask,
                first_token_idxes=first_token_idxes,
                first_token_idxes_mask=first_token_idxes_mask,
                line_rank_id=line_rank_id,
                line_rank_inner_id=line_rank_inner_id,
            )
        # SER
        if self.model_cfg.use_inner_id:
            sequence_output = torch.cat(
                (
                    text_mm_feat,
                    self.geolayoutlm_model.text_encoder.embeddings.line_rank_inner_embeddings(line_rank_inner_id)
                ),2
            )
            sequence_output = self.dropout(sequence_output)
            logits4labeling = self.bio_classifier(sequence_output) # [batch_size, seq_len, nc]
        else:
            bio_text_mm_feat = self.dropout(text_mm_feat)
            logits4labeling = self.bio_classifier(bio_text_mm_feat) # [batch_size, seq_len, nc]
        
        # RE
        batch_size, blk_len = first_token_idxes.shape
        B_batch_dim = torch.arange(0, batch_size,
            device=text_mm_feat.device).reshape(
            batch_size,1).expand(batch_size, blk_len)
        
        text_mm_blk_features = text_mm_feat[B_batch_dim, first_token_idxes]
        text_mm_blk_features = text_mm_blk_features * first_token_idxes_mask.unsqueeze(2)

        if self.model_cfg.backbone in [
            "alibaba-damo/geolayoutlm-base-uncased",
            "alibaba-damo/geolayoutlm-large-uncased",
        ]:        
            visual_mm_blk_features = vis_mm_feat[:,1:] # the global image feature; [batch_size, block_num, hidden_size]
            mixed_blk_features = self.dropout(visual_mm_blk_features + text_mm_blk_features)

            logits4linking_list = []
            logits4linking = self.pair_geometric_head(mixed_blk_features) # [batch_size, block_num, block_num]
            logits4linking_list.append(logits4linking)
            logits4linking_ref = self.multi_pairs_geometric_head(mixed_blk_features, logits4linking, first_token_idxes_mask)
            logits4linking_list.append(logits4linking_ref)

        # output and loss
        head_outputs = {
            "logits4labeling": logits4labeling,
            "logits4linking_list": logits4linking_list,
            "max_prob_as_father": self.model_cfg.max_prob_as_father,
            "max_prob_as_father_upperbound": self.model_cfg.max_prob_as_father_upperbound,
            "is_geo": self.model_cfg.backbone in [
                "alibaba-damo/geolayoutlm-base-uncased",
                "alibaba-damo/geolayoutlm-large-uncased",
            ]
        }
        head_outputs["pred4linking"] = torch.where(
            torch.sigmoid(head_outputs["logits4linking_list"][-1]) >= 0.5, \
            torch.ones_like(head_outputs["logits4linking_list"][-1]),
            torch.zeros_like(head_outputs["logits4linking_list"][-1]))
        # losses = self._get_loss(head_outputs, batch)

        # return head_outputs
        return logits4labeling

    def _get_loss(self, head_outputs, batch):
        labeling_loss, linking_loss = 0.0, 0.0
        # labeling loss
        labeling_loss = labeling_loss + self.loss_func_labeling(
            head_outputs["logits4labeling"].transpose(1, 2),
            batch["bio_labels"]
        )
        # linking loss
        for logits_lk in head_outputs["logits4linking_list"]:
            linking_loss_pairwise = self.loss_func_linking(
                logits_lk,
                batch["el_labels_blk"]
            )
            label_mask = batch["el_label_blk_mask"]
            linking_loss_all = torch.mul(linking_loss_pairwise, label_mask)
            linking_loss_all = torch.sum(linking_loss_all) / (label_mask.sum() + 1e-7)

            positive_label_mask = (batch["el_labels_blk"] > 0).float() * label_mask
            linking_loss_positive =  torch.mul(linking_loss_pairwise, positive_label_mask)
            linking_loss_positive = torch.sum(linking_loss_positive) / (positive_label_mask.sum() + 1e-7)

            # make each positive prob the same
            prob_lk = torch.sigmoid(logits_lk)
            mu_p = torch.mul(prob_lk, positive_label_mask).sum(2, keepdim=True) / (positive_label_mask.sum(2, keepdim=True) + 1e-7)
            var_p = torch.pow(((prob_lk - mu_p) * positive_label_mask), 2).sum(2) / (positive_label_mask.sum(2) + 1e-7) # [b, T]
            var_mask = (positive_label_mask.sum(2) > 1).float()
            var_p = (var_p * var_mask).sum(1) / (var_mask.sum(1) + 1e-7)
            var_p = var_p.mean()
            
            linking_loss = linking_loss + (linking_loss_all + linking_loss_positive + var_p)

        loss_dict = {
            "labeling_loss": labeling_loss,
            "linking_loss": linking_loss,
            "total_loss": labeling_loss + linking_loss,
        }
        return loss_dict


def load_model(model, state_dict):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')
    start_prefix = ''
    if not hasattr(model, 'geolayoutlm_model') and any(s.startswith('geolayoutlm_model.') for s in state_dict.keys()):
        start_prefix = 'geolayoutlm_model.'
    load(model, prefix=start_prefix)
    if len(missing_keys) > 0:
        logger.info("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        logger.info("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                           model.__class__.__name__, "\n\t".join(error_msgs)))

    return model
