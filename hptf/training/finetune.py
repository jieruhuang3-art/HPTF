"""
This script provides an exmaple to wrap UER-py for classification.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import random
import argparse
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from hptf.uer.layers import *
from hptf.uer.encoders import *
from hptf.uer.utils.vocab import Vocab
from hptf.uer.utils.constants import *
from hptf.uer.utils import *
from hptf.uer.utils.optimizers import *
from hptf.uer.utils.config import load_hyperparam
from hptf.uer.utils.seed import set_seed
from hptf.uer.model_saver import save_model
from hptf.uer.opts import finetune_opts
import tqdm
import numpy as np


def _config_to_argv(config_path):
    import yaml

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    def add(flag, value, out):
        if value is None:
            return
        if isinstance(value, bool):
            if value:
                out.append(flag)
            return
        out.extend([flag, str(value)])

    data = cfg.get("data", {})
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    argv = []
    add("--train_path", data.get("train_path"), argv)
    add("--dev_path", data.get("valid_path") or data.get("dev_path"), argv)
    add("--test_path", data.get("test_path"), argv)
    add("--vocab_path", data.get("vocab_path"), argv)
    add("--pretrained_model_path", model.get("pretrained_checkpoint"), argv)
    add("--config_path", model.get("config_path", "configs/bert_base_config.json"), argv)
    add("--classifier_arch", model.get("classifier_arch", "flowsem_mae"), argv)
    add("--use_aux_features", model.get("use_auxiliary_features", True), argv)
    add("--aux_feature_fusion", model.get("aux_feature_fusion", "gate"), argv)
    add("--flowsem_use_context_encoder", model.get("use_context_encoder", True), argv)
    add("--flowsem_cls_residual", model.get("use_contextual_residual", True), argv)
    add("--flowsem_direct_token_pooling", model.get("direct_token_pooling", True), argv)
    add("--flowsem_fusion_norm", model.get("fusion_norm", True), argv)
    add("--flowsem_branch_fusion", model.get("branch_fusion", "sum"), argv)
    add("--packet_token_layers", model.get("intra_packet_layers", 1), argv)
    add("--packet_flow_layers", model.get("inter_packet_layers", 2), argv)
    add("--flowsem_field_slots", model.get("field_slots", 32), argv)
    add("--seq_length", model.get("seq_length", model.get("max_seq_length", 128)), argv)
    add("--batch_size", training.get("batch_size"), argv)
    add("--learning_rate", training.get("learning_rate"), argv)
    add("--epochs_num", training.get("epochs", training.get("epochs_num")), argv)
    add("--warmup", training.get("warmup_ratio", training.get("warmup")), argv)
    add("--seed", training.get("seed"), argv)
    output_dir = training.get("output_dir")
    if output_dir:
        add("--output_model_path", os.path.join(output_dir, "best_model.bin"), argv)
    return argv


def _expand_config_argv():
    argv = sys.argv[1:]
    if "--config" not in argv:
        return argv
    idx = argv.index("--config")
    if idx + 1 >= len(argv):
        return argv
    config_path = argv[idx + 1]
    rest = argv[:idx] + argv[idx + 2:]
    return _config_to_argv(config_path) + rest


def read_label_counts(path):
    counts = Counter()
    with open(path, mode="r", encoding="utf-8") as f:
        columns = {}
        for line_id, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if line_id == 0:
                columns = {name: idx for idx, name in enumerate(parts)}
                continue
            counts[int(parts[columns["label"]])] += 1
    return counts


def build_class_weights(args):
    if args.class_weighting == "none":
        return None
    if args.class_weighting == "legacy_5class":
        if args.labels_num == 5:
            return [1.0, 1.0, 3.0, 3.0, 5.0]
        return None

    counts = read_label_counts(args.train_path)
    total = sum(counts.values())
    weights = []
    for label_id in range(args.labels_num):
        label_count = max(counts.get(label_id, 0), 1)
        weights.append(total / float(args.labels_num * label_count))
    if args.class_weighting == "sqrt_balanced":
        weights = [weight ** 0.5 for weight in weights]
    mean_weight = sum(weights) / len(weights)
    return [weight / mean_weight for weight in weights]


def configure_loss(module, args):
    module.focal_loss_gamma = args.focal_loss_gamma
    if args.class_weights is None:
        module.class_weights = None
    else:
        module.register_buffer("class_weights", torch.tensor(args.class_weights, dtype=torch.float))


def classification_loss(logits, tgt, class_weights=None, focal_loss_gamma=0.0):
    targets = tgt.view(-1)
    weights = class_weights.to(logits.device) if class_weights is not None else None
    if focal_loss_gamma > 0.0:
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = target_log_probs.exp()
        ce = F.nll_loss(log_probs, targets, weight=weights, reduction="none")
        return (((1.0 - pt) ** focal_loss_gamma) * ce).mean()
    return F.cross_entropy(logits, targets, weight=weights)


class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)
        self.hierarchical_pooling = args.hierarchical_pooling
        self.use_aux_features = args.use_aux_features
        self.aux_feature_fusion = args.aux_feature_fusion
        configure_loss(self, args)
        if self.hierarchical_pooling == "packet_attention":
            self.packet_attention = nn.Linear(args.hidden_size, 1)
        if self.use_aux_features:
            self.time_feature_proj = nn.Linear(1, args.hidden_size)
            self.length_feature_proj = nn.Linear(1, args.hidden_size)
            self.aux_feature_norm = nn.LayerNorm(args.hidden_size)
            if self.aux_feature_fusion == "gate":
                self.aux_gate = nn.Linear(args.hidden_size * 2, args.hidden_size)
            elif self.aux_feature_fusion == "cross_attention":
                self.aux_query_proj = nn.Linear(args.hidden_size, args.hidden_size)
                self.aux_key_proj = nn.Linear(args.hidden_size, args.hidden_size)
                self.aux_value_proj = nn.Linear(args.hidden_size, args.hidden_size)
                self.aux_out_proj = nn.Linear(args.hidden_size, args.hidden_size)

    def _unpack_encoder_output(self, encoder_output):
        if isinstance(encoder_output, tuple):
            return encoder_output
        return encoder_output, {}

    def _cross_attention_fuse(self, emb, aux_emb, seg):
        query = self.aux_query_proj(emb)
        key = self.aux_key_proj(aux_emb)
        value = self.aux_value_proj(aux_emb)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)

        padding_mask = (seg > 0).unsqueeze(1).expand(-1, seg.size(1), -1)
        scores = scores.masked_fill(~padding_mask, -1e4)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, value)
        context = self.aux_out_proj(context)
        return emb + context * (seg > 0).unsqueeze(-1).float()

    def _packet_representations(self, output, seg, packet_start_mask):
        batch_size, seq_length, hidden_size = output.size()
        valid_mask = seg > 0
        packet_reps = []
        packet_valid_masks = []
        max_packets = 1

        for batch_idx in range(batch_size):
            starts = torch.nonzero(packet_start_mask[batch_idx] > 0, as_tuple=False).flatten().tolist()
            sample_valid = valid_mask[batch_idx]
            token_limit = int(sample_valid.long().sum().item())

            sample_packets = []
            for start_idx, start_pos in enumerate(starts):
                if start_pos >= token_limit:
                    continue
                end_pos = token_limit
                if start_idx + 1 < len(starts):
                    end_pos = min(starts[start_idx + 1], token_limit)
                if end_pos <= start_pos + 1:
                    packet_hidden = output[batch_idx, start_pos:end_pos].mean(dim=0)
                else:
                    packet_hidden = output[batch_idx, start_pos + 1:end_pos].mean(dim=0)
                sample_packets.append(packet_hidden)

            if not sample_packets:
                cls_pos = 0
                sample_packets = [output[batch_idx, cls_pos]]

            max_packets = max(max_packets, len(sample_packets))
            packet_reps.append(sample_packets)

        device = output.device
        packet_tensor = output.new_zeros(batch_size, max_packets, hidden_size)
        packet_valid = torch.zeros(batch_size, max_packets, dtype=torch.bool, device=device)

        for batch_idx, sample_packets in enumerate(packet_reps):
            for packet_idx, packet_hidden in enumerate(sample_packets):
                packet_tensor[batch_idx, packet_idx] = packet_hidden
                packet_valid[batch_idx, packet_idx] = True

        return packet_tensor, packet_valid

    def _pool_sequence(self, output, seg, packet_start_mask=None):
        if self.hierarchical_pooling != "none" and packet_start_mask is not None:
            packet_reps, packet_valid = self._packet_representations(output, seg, packet_start_mask)
            if self.hierarchical_pooling == "packet_mean":
                masked = packet_reps * packet_valid.unsqueeze(-1).float()
                return masked.sum(dim=1) / packet_valid.sum(dim=1, keepdim=True).clamp_min(1.0)

            attn_logits = self.packet_attention(packet_reps).squeeze(-1)
            attn_logits = attn_logits.masked_fill(~packet_valid, -1e4)
            attn = F.softmax(attn_logits, dim=-1)
            return torch.sum(packet_reps * attn.unsqueeze(-1), dim=1)

        if self.pooling == "mean":
            return torch.mean(output, dim=1)
        if self.pooling == "max":
            return torch.max(output, dim=1)[0]
        if self.pooling == "last":
            return output[:, -1, :]
        return output[:, 0, :]

    def forward(self, src, tgt, seg, soft_tgt=None, time_delta=None, pkt_len=None, packet_start_mask=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        if self.use_aux_features and time_delta is not None and pkt_len is not None:
            time_emb = self.time_feature_proj(time_delta.unsqueeze(-1))
            len_emb = self.length_feature_proj(pkt_len.unsqueeze(-1))
            aux_emb = self.aux_feature_norm(time_emb + len_emb)
            non_padding = (seg > 0).unsqueeze(-1).float()
            aux_emb = aux_emb * non_padding
            if self.aux_feature_fusion == "add":
                emb = emb + aux_emb
            elif self.aux_feature_fusion == "cross_attention":
                emb = self._cross_attention_fuse(emb, aux_emb, seg)
            else:
                gate = torch.sigmoid(self.aux_gate(torch.cat([emb, aux_emb], dim=-1)))
                emb = emb + gate * aux_emb
        # Encoder.
        encoder_output = self.encoder(emb, seg)
        output, _ = self._unpack_encoder_output(encoder_output)
        pooled = self._pool_sequence(output, seg, packet_start_mask)
        pooled = torch.tanh(self.output_layer_1(pooled))
        logits = self.output_layer_2(pooled)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = classification_loss(logits, tgt, self.class_weights, self.focal_loss_gamma)
            return loss, logits
        else:
            return None, logits


class PacketFlowTransformerClassifier(nn.Module):
    def __init__(self, args):
        super(PacketFlowTransformerClassifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.labels_num = args.labels_num
        self.use_aux_features = args.use_aux_features
        self.aux_feature_fusion = args.aux_feature_fusion
        configure_loss(self, args)

        if self.use_aux_features:
            self.time_feature_proj = nn.Linear(1, args.hidden_size)
            self.length_feature_proj = nn.Linear(1, args.hidden_size)
            self.aux_feature_norm = nn.LayerNorm(args.hidden_size)
            if self.aux_feature_fusion == "cross_attention":
                self.aux_query_proj = nn.Linear(args.hidden_size, args.hidden_size)
                self.aux_key_proj = nn.Linear(args.hidden_size, args.hidden_size)
                self.aux_value_proj = nn.Linear(args.hidden_size, args.hidden_size)
                self.aux_out_proj = nn.Linear(args.hidden_size, args.hidden_size)
            else:
                self.aux_gate = nn.Linear(args.hidden_size * 2, args.hidden_size)

        self.packet_token_attention = nn.Linear(args.hidden_size, 1)
        self.packet_pos_embedding = nn.Embedding(args.seq_length, args.hidden_size)

        packet_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_size,
            nhead=args.heads_num,
            dim_feedforward=args.feedforward_size,
            dropout=args.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.packet_flow_encoder = nn.TransformerEncoder(packet_layer, num_layers=args.packet_flow_layers)
        self.flow_attention = nn.Linear(args.hidden_size, 1)
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def _cross_attention_fuse(self, emb, aux_emb, seg):
        query = self.aux_query_proj(emb)
        key = self.aux_key_proj(aux_emb)
        value = self.aux_value_proj(aux_emb)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        padding_mask = (seg > 0).unsqueeze(1).expand(-1, seg.size(1), -1)
        scores = scores.masked_fill(~padding_mask, -1e4)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, value)
        context = self.aux_out_proj(context)
        return emb + context * (seg > 0).unsqueeze(-1).float()

    def _fuse_aux_features(self, emb, seg, time_delta, pkt_len):
        if not self.use_aux_features or time_delta is None or pkt_len is None:
            return emb
        time_emb = self.time_feature_proj(time_delta.unsqueeze(-1))
        len_emb = self.length_feature_proj(pkt_len.unsqueeze(-1))
        aux_emb = self.aux_feature_norm(time_emb + len_emb)
        non_padding = (seg > 0).unsqueeze(-1).float()
        aux_emb = aux_emb * non_padding
        if self.aux_feature_fusion == "add":
            return emb + aux_emb
        if self.aux_feature_fusion == "cross_attention":
            return self._cross_attention_fuse(emb, aux_emb, seg)
        gate = torch.sigmoid(self.aux_gate(torch.cat([emb, aux_emb], dim=-1)))
        return emb + gate * aux_emb

    def _packet_representations(self, emb, seg, packet_start_mask):
        batch_size, seq_length, hidden_size = emb.size()
        valid_mask = seg > 0
        packet_reps = []
        max_packets = 1

        for batch_idx in range(batch_size):
            starts = []
            if packet_start_mask is not None:
                starts = torch.nonzero(packet_start_mask[batch_idx] > 0, as_tuple=False).flatten().tolist()
            token_limit = int(valid_mask[batch_idx].long().sum().item())
            sample_packets = []

            for start_idx, start_pos in enumerate(starts):
                if start_pos >= token_limit:
                    continue
                end_pos = token_limit
                if start_idx + 1 < len(starts):
                    end_pos = min(starts[start_idx + 1], token_limit)
                if end_pos <= start_pos:
                    continue
                token_hidden = emb[batch_idx, start_pos:end_pos]
                token_scores = self.packet_token_attention(token_hidden).squeeze(-1)
                token_attn = F.softmax(token_scores, dim=-1)
                sample_packets.append(torch.sum(token_hidden * token_attn.unsqueeze(-1), dim=0))

            if not sample_packets:
                token_hidden = emb[batch_idx, :max(token_limit, 1)]
                token_scores = self.packet_token_attention(token_hidden).squeeze(-1)
                token_attn = F.softmax(token_scores, dim=-1)
                sample_packets = [torch.sum(token_hidden * token_attn.unsqueeze(-1), dim=0)]

            max_packets = max(max_packets, len(sample_packets))
            packet_reps.append(sample_packets)

        packet_tensor = emb.new_zeros(batch_size, max_packets, hidden_size)
        packet_valid = torch.zeros(batch_size, max_packets, dtype=torch.bool, device=emb.device)
        for batch_idx, sample_packets in enumerate(packet_reps):
            for packet_idx, packet_hidden in enumerate(sample_packets):
                packet_tensor[batch_idx, packet_idx] = packet_hidden
                packet_valid[batch_idx, packet_idx] = True

        return packet_tensor, packet_valid

    def forward(self, src, tgt, seg, soft_tgt=None, time_delta=None, pkt_len=None, packet_start_mask=None):
        emb = self.embedding(src, seg)
        emb = self._fuse_aux_features(emb, seg, time_delta, pkt_len)

        packet_tensor, packet_valid = self._packet_representations(emb, seg, packet_start_mask)
        positions = torch.arange(packet_tensor.size(1), device=packet_tensor.device).unsqueeze(0)
        packet_tensor = packet_tensor + self.packet_pos_embedding(positions)
        encoded_packets = self.packet_flow_encoder(packet_tensor, src_key_padding_mask=~packet_valid)

        attn_logits = self.flow_attention(encoded_packets).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~packet_valid, -1e4)
        attn = F.softmax(attn_logits, dim=-1)
        pooled = torch.sum(encoded_packets * attn.unsqueeze(-1), dim=1)
        pooled = torch.tanh(self.output_layer_1(pooled))
        logits = self.output_layer_2(pooled)

        if tgt is not None:
            loss = classification_loss(logits, tgt, self.class_weights, self.focal_loss_gamma)
            return loss, logits
        return None, logits


class PacketContextFlowTransformerClassifier(nn.Module):
    def __init__(self, args):
        super(PacketContextFlowTransformerClassifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.labels_num = args.labels_num
        self.use_aux_features = args.use_aux_features
        self.aux_feature_fusion = args.aux_feature_fusion
        configure_loss(self, args)

        if self.use_aux_features:
            self.time_feature_proj = nn.Linear(1, args.hidden_size)
            self.length_feature_proj = nn.Linear(1, args.hidden_size)
            self.aux_feature_norm = nn.LayerNorm(args.hidden_size)
            if self.aux_feature_fusion == "cross_attention":
                self.aux_query_proj = nn.Linear(args.hidden_size, args.hidden_size)
                self.aux_key_proj = nn.Linear(args.hidden_size, args.hidden_size)
                self.aux_value_proj = nn.Linear(args.hidden_size, args.hidden_size)
                self.aux_out_proj = nn.Linear(args.hidden_size, args.hidden_size)
            else:
                self.aux_gate = nn.Linear(args.hidden_size * 2, args.hidden_size)

        token_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_size,
            nhead=args.heads_num,
            dim_feedforward=args.feedforward_size,
            dropout=args.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.token_context_encoder = nn.TransformerEncoder(
            token_layer,
            num_layers=args.packet_token_layers,
            enable_nested_tensor=False,
        )
        self.token_context_norm = nn.LayerNorm(args.hidden_size)

        self.packet_token_attention = nn.Linear(args.hidden_size, 1)
        self.packet_pos_embedding = nn.Embedding(args.seq_length, args.hidden_size)
        packet_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_size,
            nhead=args.heads_num,
            dim_feedforward=args.feedforward_size,
            dropout=args.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.packet_flow_encoder = nn.TransformerEncoder(packet_layer, num_layers=args.packet_flow_layers)
        self.flow_attention = nn.Linear(args.hidden_size, 1)
        self.direct_packet_attention = nn.Linear(args.hidden_size, 1)
        self.flow_stats_proj = nn.Sequential(
            nn.Linear(4, args.hidden_size),
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.hidden_size),
        )
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def _cross_attention_fuse(self, emb, aux_emb, seg):
        query = self.aux_query_proj(emb)
        key = self.aux_key_proj(aux_emb)
        value = self.aux_value_proj(aux_emb)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        padding_mask = (seg > 0).unsqueeze(1).expand(-1, seg.size(1), -1)
        scores = scores.masked_fill(~padding_mask, -1e4)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, value)
        context = self.aux_out_proj(context)
        return emb + context * (seg > 0).unsqueeze(-1).float()

    def _fuse_aux_features(self, emb, seg, time_delta, pkt_len):
        if not self.use_aux_features or time_delta is None or pkt_len is None:
            return emb
        time_emb = self.time_feature_proj(time_delta.unsqueeze(-1))
        len_emb = self.length_feature_proj(pkt_len.unsqueeze(-1))
        aux_emb = self.aux_feature_norm(time_emb + len_emb)
        non_padding = (seg > 0).unsqueeze(-1).float()
        aux_emb = aux_emb * non_padding
        if self.aux_feature_fusion == "add":
            return emb + aux_emb
        if self.aux_feature_fusion == "cross_attention":
            return self._cross_attention_fuse(emb, aux_emb, seg)
        gate = torch.sigmoid(self.aux_gate(torch.cat([emb, aux_emb], dim=-1)))
        return emb + gate * aux_emb

    def _flow_stats(self, seg, time_delta, pkt_len):
        valid = (seg > 0).float()
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        if time_delta is None or pkt_len is None:
            return valid.new_zeros(valid.size(0), 4)

        time_mean = (time_delta * valid).sum(dim=1, keepdim=True) / denom
        len_mean = (pkt_len * valid).sum(dim=1, keepdim=True) / denom
        time_var = (((time_delta - time_mean) * valid) ** 2).sum(dim=1, keepdim=True) / denom
        len_var = (((pkt_len - len_mean) * valid) ** 2).sum(dim=1, keepdim=True) / denom
        return torch.cat([time_mean, time_var.sqrt(), len_mean, len_var.sqrt()], dim=1)

    def _packet_representations(self, output, seg, packet_start_mask):
        batch_size, seq_length, hidden_size = output.size()
        valid_mask = seg > 0
        packet_reps = []
        max_packets = 1

        for batch_idx in range(batch_size):
            starts = []
            if packet_start_mask is not None:
                starts = torch.nonzero(packet_start_mask[batch_idx] > 0, as_tuple=False).flatten().tolist()
            token_limit = int(valid_mask[batch_idx].long().sum().item())
            sample_packets = []

            for start_idx, start_pos in enumerate(starts):
                if start_pos >= token_limit:
                    continue
                end_pos = token_limit
                if start_idx + 1 < len(starts):
                    end_pos = min(starts[start_idx + 1], token_limit)
                if end_pos <= start_pos:
                    continue
                token_hidden = output[batch_idx, start_pos:end_pos]
                token_scores = self.packet_token_attention(token_hidden).squeeze(-1)
                token_attn = F.softmax(token_scores, dim=-1)
                sample_packets.append(torch.sum(token_hidden * token_attn.unsqueeze(-1), dim=0))

            if not sample_packets:
                token_hidden = output[batch_idx, :max(token_limit, 1)]
                token_scores = self.packet_token_attention(token_hidden).squeeze(-1)
                token_attn = F.softmax(token_scores, dim=-1)
                sample_packets = [torch.sum(token_hidden * token_attn.unsqueeze(-1), dim=0)]

            max_packets = max(max_packets, len(sample_packets))
            packet_reps.append(sample_packets)

        packet_tensor = output.new_zeros(batch_size, max_packets, hidden_size)
        packet_valid = torch.zeros(batch_size, max_packets, dtype=torch.bool, device=output.device)
        for batch_idx, sample_packets in enumerate(packet_reps):
            for packet_idx, packet_hidden in enumerate(sample_packets):
                packet_tensor[batch_idx, packet_idx] = packet_hidden
                packet_valid[batch_idx, packet_idx] = True
        return packet_tensor, packet_valid

    def forward(self, src, tgt, seg, soft_tgt=None, time_delta=None, pkt_len=None, packet_start_mask=None):
        emb = self.embedding(src, seg)
        emb = self._fuse_aux_features(emb, seg, time_delta, pkt_len)

        token_padding = ~(seg > 0)
        token_context = self.token_context_encoder(emb, src_key_padding_mask=token_padding)
        token_context = self.token_context_norm(emb + token_context)

        packet_tensor, packet_valid = self._packet_representations(token_context, seg, packet_start_mask)
        positions = torch.arange(packet_tensor.size(1), device=packet_tensor.device).unsqueeze(0)
        packet_tensor = packet_tensor + self.packet_pos_embedding(positions)
        encoded_packets = self.packet_flow_encoder(packet_tensor, src_key_padding_mask=~packet_valid)

        flow_logits = self.flow_attention(encoded_packets).squeeze(-1)
        flow_logits = flow_logits.masked_fill(~packet_valid, -1e4)
        flow_attn = F.softmax(flow_logits, dim=-1)
        flow_pooled = torch.sum(encoded_packets * flow_attn.unsqueeze(-1), dim=1)

        direct_logits = self.direct_packet_attention(packet_tensor).squeeze(-1)
        direct_logits = direct_logits.masked_fill(~packet_valid, -1e4)
        direct_attn = F.softmax(direct_logits, dim=-1)
        direct_pooled = torch.sum(packet_tensor * direct_attn.unsqueeze(-1), dim=1)

        stats = self.flow_stats_proj(self._flow_stats(seg, time_delta, pkt_len))
        pooled = flow_pooled + direct_pooled + stats
        pooled = torch.tanh(self.output_layer_1(pooled))
        logits = self.output_layer_2(pooled)

        if tgt is not None:
            loss = classification_loss(logits, tgt, self.class_weights, self.focal_loss_gamma)
            return loss, logits
        return None, logits


class FlowSemMAEClassifier(nn.Module):
    def __init__(self, args):
        super(FlowSemMAEClassifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args) if args.flowsem_use_context_encoder else None
        self.labels_num = args.labels_num
        self.use_aux_features = args.use_aux_features
        self.aux_feature_fusion = args.aux_feature_fusion
        self.field_slots = args.flowsem_field_slots
        self.direct_token_pooling = args.flowsem_direct_token_pooling
        self.use_context_encoder = args.flowsem_use_context_encoder
        self.cls_residual = args.flowsem_cls_residual
        self.fusion_norm_enabled = args.flowsem_fusion_norm
        self.branch_fusion = args.flowsem_branch_fusion
        self.no_dual_axis_transformer = getattr(args, "flowsem_no_dual_axis_transformer", False)
        self.no_field_axis_transformer = getattr(args, "flowsem_no_field_axis_transformer", False)
        self.no_temporal_transformer = getattr(args, "flowsem_no_temporal_transformer", False)
        configure_loss(self, args)

        if self.use_aux_features:
            self.time_feature_proj = nn.Linear(1, args.hidden_size)
            self.length_feature_proj = nn.Linear(1, args.hidden_size)
            self.aux_feature_norm = nn.LayerNorm(args.hidden_size)
            if self.aux_feature_fusion == "cross_attention":
                self.aux_query_proj = nn.Linear(args.hidden_size, args.hidden_size)
                self.aux_key_proj = nn.Linear(args.hidden_size, args.hidden_size)
                self.aux_value_proj = nn.Linear(args.hidden_size, args.hidden_size)
                self.aux_out_proj = nn.Linear(args.hidden_size, args.hidden_size)
            else:
                self.aux_gate = nn.Linear(args.hidden_size * 2, args.hidden_size)

        field_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_size,
            nhead=args.heads_num,
            dim_feedforward=args.feedforward_size,
            dropout=args.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.field_encoder = nn.TransformerEncoder(
            field_layer,
            num_layers=args.packet_token_layers,
            enable_nested_tensor=False,
        )
        self.field_pos_embedding = nn.Embedding(self.field_slots, args.hidden_size)
        self.field_attention = nn.Linear(args.hidden_size, 1)

        packet_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_size,
            nhead=args.heads_num,
            dim_feedforward=args.feedforward_size,
            dropout=args.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            packet_layer,
            num_layers=args.packet_flow_layers,
            enable_nested_tensor=False,
        )
        self.packet_pos_embedding = nn.Embedding(args.seq_length, args.hidden_size)
        self.packet_attention = nn.Linear(args.hidden_size, 1)
        if self.direct_token_pooling:
            self.direct_token_attention = nn.Linear(args.hidden_size, 1)
        self.flow_stats_proj = nn.Sequential(
            nn.Linear(4, args.hidden_size),
            nn.Tanh(),
            nn.Linear(args.hidden_size, args.hidden_size),
        )
        if self.cls_residual:
            self.cls_fusion = nn.Sequential(
                nn.Linear(args.hidden_size * 2, args.hidden_size),
                nn.Tanh(),
                nn.Linear(args.hidden_size, args.hidden_size),
            )
        if self.branch_fusion == "gate":
            self.branch_gate = nn.Linear(args.hidden_size, 1)
        elif self.branch_fusion == "concat":
            self.branch_fusion_proj = nn.Sequential(
                nn.Linear(args.hidden_size * 4, args.hidden_size),
                nn.Tanh(),
                nn.Linear(args.hidden_size, args.hidden_size),
            )
        if self.fusion_norm_enabled:
            self.fusion_norm = nn.LayerNorm(args.hidden_size)
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def _unpack_encoder_output(self, encoder_output):
        if isinstance(encoder_output, tuple):
            return encoder_output
        return encoder_output, {}

    def _cross_attention_fuse(self, emb, aux_emb, seg):
        query = self.aux_query_proj(emb)
        key = self.aux_key_proj(aux_emb)
        value = self.aux_value_proj(aux_emb)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        padding_mask = (seg > 0).unsqueeze(1).expand(-1, seg.size(1), -1)
        scores = scores.masked_fill(~padding_mask, -1e4)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, value)
        context = self.aux_out_proj(context)
        return emb + context * (seg > 0).unsqueeze(-1).float()

    def _fuse_aux_features(self, emb, seg, time_delta, pkt_len):
        if not self.use_aux_features or time_delta is None or pkt_len is None:
            return emb
        time_emb = self.time_feature_proj(time_delta.unsqueeze(-1))
        len_emb = self.length_feature_proj(pkt_len.unsqueeze(-1))
        aux_emb = self.aux_feature_norm(time_emb + len_emb)
        aux_emb = aux_emb * (seg > 0).unsqueeze(-1).float()
        if self.aux_feature_fusion == "add":
            return emb + aux_emb
        if self.aux_feature_fusion == "cross_attention":
            return self._cross_attention_fuse(emb, aux_emb, seg)
        gate = torch.sigmoid(self.aux_gate(torch.cat([emb, aux_emb], dim=-1)))
        return emb + gate * aux_emb

    def _flow_stats(self, seg, time_delta, pkt_len):
        valid = (seg > 0).float()
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        if time_delta is None or pkt_len is None:
            return valid.new_zeros(valid.size(0), 4)

        time_mean = (time_delta * valid).sum(dim=1, keepdim=True) / denom
        len_mean = (pkt_len * valid).sum(dim=1, keepdim=True) / denom
        time_var = (((time_delta - time_mean) * valid) ** 2).sum(dim=1, keepdim=True) / denom
        len_var = (((pkt_len - len_mean) * valid) ** 2).sum(dim=1, keepdim=True) / denom
        return torch.cat([time_mean, time_var.sqrt(), len_mean, len_var.sqrt()], dim=1)

    def _flowsem_table(self, emb, seg, packet_start_mask):
        batch_size, _, hidden_size = emb.size()
        valid_mask = seg > 0
        samples = []
        max_packets = 1

        for batch_idx in range(batch_size):
            token_limit = int(valid_mask[batch_idx].long().sum().item())
            starts = []
            if packet_start_mask is not None:
                starts = torch.nonzero(packet_start_mask[batch_idx] > 0, as_tuple=False).flatten().tolist()
            starts = [start for start in starts if start < token_limit]
            if not starts:
                starts = [0]

            packet_slots = []
            for start_idx, start_pos in enumerate(starts):
                end_pos = token_limit
                if start_idx + 1 < len(starts):
                    end_pos = min(starts[start_idx + 1], token_limit)
                if end_pos <= start_pos:
                    continue
                token_hidden = emb[batch_idx, start_pos:end_pos]
                field_count = min(token_hidden.size(0), self.field_slots)
                slot = emb.new_zeros(self.field_slots, hidden_size)
                valid = torch.zeros(self.field_slots, dtype=torch.bool, device=emb.device)
                if field_count > 0:
                    slot[:field_count] = token_hidden[:field_count]
                    valid[:field_count] = True
                packet_slots.append((slot, valid))

            if not packet_slots:
                slot = emb.new_zeros(self.field_slots, hidden_size)
                valid = torch.zeros(self.field_slots, dtype=torch.bool, device=emb.device)
                slot[0] = emb[batch_idx, 0]
                valid[0] = True
                packet_slots.append((slot, valid))

            max_packets = max(max_packets, len(packet_slots))
            samples.append(packet_slots)

        table = emb.new_zeros(batch_size, max_packets, self.field_slots, hidden_size)
        field_valid = torch.zeros(batch_size, max_packets, self.field_slots, dtype=torch.bool, device=emb.device)
        packet_valid = torch.zeros(batch_size, max_packets, dtype=torch.bool, device=emb.device)
        for batch_idx, packet_slots in enumerate(samples):
            for packet_idx, (slot, valid) in enumerate(packet_slots):
                table[batch_idx, packet_idx] = slot
                field_valid[batch_idx, packet_idx] = valid
                packet_valid[batch_idx, packet_idx] = True
        return table, field_valid, packet_valid

    def forward(self, src, tgt, seg, soft_tgt=None, time_delta=None, pkt_len=None, packet_start_mask=None):
        emb = self.embedding(src, seg)
        emb = self._fuse_aux_features(emb, seg, time_delta, pkt_len)

        cls_pooled = None
        features = emb
        if self.use_context_encoder:
            encoder_output = self.encoder(emb, seg)
            features, _ = self._unpack_encoder_output(encoder_output)
            cls_pooled = features[:, 0, :]

        direct_pooled = None
        if self.direct_token_pooling:
            token_valid = seg > 0
            token_logits = self.direct_token_attention(features).squeeze(-1)
            token_logits = token_logits.masked_fill(~token_valid, -1e4)
            token_attn = F.softmax(token_logits, dim=-1)
            direct_pooled = torch.sum(features * token_attn.unsqueeze(-1), dim=1)

        table, field_valid, packet_valid = self._flowsem_table(features, seg, packet_start_mask)
        batch_size, packets_num, fields_num, hidden_size = table.size()
        field_positions = torch.arange(fields_num, device=table.device).view(1, 1, fields_num)
        table = table + self.field_pos_embedding(field_positions)

        field_input = table.reshape(batch_size * packets_num, fields_num, hidden_size)
        field_mask = ~field_valid.reshape(batch_size * packets_num, fields_num)
        empty_rows = field_mask.all(dim=1)
        if empty_rows.any():
            field_mask[empty_rows, 0] = False
        if self.no_dual_axis_transformer or self.no_field_axis_transformer:
            field_valid_flat = field_valid.reshape(batch_size * packets_num, fields_num).float()
            denom = field_valid_flat.sum(dim=1, keepdim=True).clamp_min(1.0)
            packet_reps = torch.sum(field_input * field_valid_flat.unsqueeze(-1), dim=1) / denom
            packet_reps = packet_reps.reshape(batch_size, packets_num, hidden_size)
            if self.no_dual_axis_transformer:
                encoded_packets = packet_reps
            else:
                packet_positions = torch.arange(packets_num, device=packet_reps.device).view(1, packets_num)
                packet_reps = packet_reps + self.packet_pos_embedding(packet_positions)
                encoded_packets = self.temporal_encoder(packet_reps, src_key_padding_mask=~packet_valid)
        else:
            encoded_fields = self.field_encoder(field_input, src_key_padding_mask=field_mask)
            field_logits = self.field_attention(encoded_fields).squeeze(-1)
            field_logits = field_logits.masked_fill(~field_valid.reshape(batch_size * packets_num, fields_num), -1e4)
            field_attn = F.softmax(field_logits, dim=-1)
            packet_reps = torch.sum(encoded_fields * field_attn.unsqueeze(-1), dim=1)
            packet_reps = packet_reps.reshape(batch_size, packets_num, hidden_size)

            if self.no_temporal_transformer:
                encoded_packets = packet_reps
            else:
                packet_positions = torch.arange(packets_num, device=packet_reps.device).view(1, packets_num)
                packet_reps = packet_reps + self.packet_pos_embedding(packet_positions)
                encoded_packets = self.temporal_encoder(packet_reps, src_key_padding_mask=~packet_valid)

        packet_logits = self.packet_attention(encoded_packets).squeeze(-1)
        packet_logits = packet_logits.masked_fill(~packet_valid, -1e4)
        packet_attn = F.softmax(packet_logits, dim=-1)
        packet_pooled = torch.sum(encoded_packets * packet_attn.unsqueeze(-1), dim=1)
        stats_pooled = self.flow_stats_proj(self._flow_stats(seg, time_delta, pkt_len))
        cls_branch = None
        if self.cls_residual and cls_pooled is not None:
            cls_base = packet_pooled + stats_pooled
            if direct_pooled is not None:
                cls_base = cls_base + direct_pooled
            cls_branch = self.cls_fusion(torch.cat([cls_base, cls_pooled], dim=-1))

        if self.branch_fusion == "gate":
            branches = [packet_pooled, stats_pooled]
            if direct_pooled is not None:
                branches.append(direct_pooled)
            if cls_branch is not None:
                branches.append(cls_branch)
            branch_tensor = torch.stack(branches, dim=1)
            branch_weights = F.softmax(self.branch_gate(branch_tensor).squeeze(-1), dim=-1)
            pooled = torch.sum(branch_tensor * branch_weights.unsqueeze(-1), dim=1)
        elif self.branch_fusion == "concat":
            zero = torch.zeros_like(packet_pooled)
            pooled = self.branch_fusion_proj(torch.cat([
                packet_pooled,
                direct_pooled if direct_pooled is not None else zero,
                stats_pooled,
                cls_branch if cls_branch is not None else zero,
            ], dim=-1))
        else:
            pooled = packet_pooled
            if direct_pooled is not None:
                pooled = pooled + direct_pooled
            pooled = pooled + stats_pooled
            if cls_branch is not None:
                pooled = pooled + cls_branch
        if self.fusion_norm_enabled:
            pooled = self.fusion_norm(pooled)
        pooled = torch.tanh(self.output_layer_1(pooled))
        logits = self.output_layer_2(pooled)

        if tgt is not None:
            loss = classification_loss(logits, tgt, self.class_weights, self.focal_loss_gamma)
            return loss, logits
        return None, logits


class HybridPacketFlowTransformerClassifier(nn.Module):
    def __init__(self, args):
        super(HybridPacketFlowTransformerClassifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.use_aux_features = args.use_aux_features
        self.aux_feature_fusion = args.aux_feature_fusion
        configure_loss(self, args)

        if self.use_aux_features:
            self.time_feature_proj = nn.Linear(1, args.hidden_size)
            self.length_feature_proj = nn.Linear(1, args.hidden_size)
            self.aux_feature_norm = nn.LayerNorm(args.hidden_size)
            if self.aux_feature_fusion == "cross_attention":
                self.aux_query_proj = nn.Linear(args.hidden_size, args.hidden_size)
                self.aux_key_proj = nn.Linear(args.hidden_size, args.hidden_size)
                self.aux_value_proj = nn.Linear(args.hidden_size, args.hidden_size)
                self.aux_out_proj = nn.Linear(args.hidden_size, args.hidden_size)
            else:
                self.aux_gate = nn.Linear(args.hidden_size * 2, args.hidden_size)

        self.packet_attention = nn.Linear(args.hidden_size, 1)
        self.packet_pos_embedding = nn.Embedding(args.seq_length, args.hidden_size)
        packet_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_size,
            nhead=args.heads_num,
            dim_feedforward=args.feedforward_size,
            dropout=args.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.packet_flow_encoder = nn.TransformerEncoder(packet_layer, num_layers=args.packet_flow_layers)
        self.flow_attention = nn.Linear(args.hidden_size, 1)
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def _unpack_encoder_output(self, encoder_output):
        if isinstance(encoder_output, tuple):
            return encoder_output
        return encoder_output, {}

    def _cross_attention_fuse(self, emb, aux_emb, seg):
        query = self.aux_query_proj(emb)
        key = self.aux_key_proj(aux_emb)
        value = self.aux_value_proj(aux_emb)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
        padding_mask = (seg > 0).unsqueeze(1).expand(-1, seg.size(1), -1)
        scores = scores.masked_fill(~padding_mask, -1e4)
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, value)
        context = self.aux_out_proj(context)
        return emb + context * (seg > 0).unsqueeze(-1).float()

    def _fuse_aux_features(self, emb, seg, time_delta, pkt_len):
        if not self.use_aux_features or time_delta is None or pkt_len is None:
            return emb
        time_emb = self.time_feature_proj(time_delta.unsqueeze(-1))
        len_emb = self.length_feature_proj(pkt_len.unsqueeze(-1))
        aux_emb = self.aux_feature_norm(time_emb + len_emb)
        aux_emb = aux_emb * (seg > 0).unsqueeze(-1).float()
        if self.aux_feature_fusion == "add":
            return emb + aux_emb
        if self.aux_feature_fusion == "cross_attention":
            return self._cross_attention_fuse(emb, aux_emb, seg)
        gate = torch.sigmoid(self.aux_gate(torch.cat([emb, aux_emb], dim=-1)))
        return emb + gate * aux_emb

    def _packet_representations(self, output, seg, packet_start_mask):
        batch_size, seq_length, hidden_size = output.size()
        valid_mask = seg > 0
        packet_reps = []
        max_packets = 1

        for batch_idx in range(batch_size):
            starts = torch.nonzero(packet_start_mask[batch_idx] > 0, as_tuple=False).flatten().tolist()
            token_limit = int(valid_mask[batch_idx].long().sum().item())
            sample_packets = []

            for start_idx, start_pos in enumerate(starts):
                if start_pos >= token_limit:
                    continue
                end_pos = token_limit
                if start_idx + 1 < len(starts):
                    end_pos = min(starts[start_idx + 1], token_limit)
                if end_pos <= start_pos + 1:
                    packet_hidden = output[batch_idx, start_pos:end_pos].mean(dim=0)
                else:
                    packet_hidden = output[batch_idx, start_pos + 1:end_pos].mean(dim=0)
                sample_packets.append(packet_hidden)

            if not sample_packets:
                sample_packets = [output[batch_idx, 0]]

            max_packets = max(max_packets, len(sample_packets))
            packet_reps.append(sample_packets)

        packet_tensor = output.new_zeros(batch_size, max_packets, hidden_size)
        packet_valid = torch.zeros(batch_size, max_packets, dtype=torch.bool, device=output.device)
        for batch_idx, sample_packets in enumerate(packet_reps):
            for packet_idx, packet_hidden in enumerate(sample_packets):
                packet_tensor[batch_idx, packet_idx] = packet_hidden
                packet_valid[batch_idx, packet_idx] = True
        return packet_tensor, packet_valid

    def forward(self, src, tgt, seg, soft_tgt=None, time_delta=None, pkt_len=None, packet_start_mask=None):
        emb = self.embedding(src, seg)
        emb = self._fuse_aux_features(emb, seg, time_delta, pkt_len)
        encoder_output = self.encoder(emb, seg)
        output, _ = self._unpack_encoder_output(encoder_output)

        packet_tensor, packet_valid = self._packet_representations(output, seg, packet_start_mask)
        positions = torch.arange(packet_tensor.size(1), device=packet_tensor.device).unsqueeze(0)
        packet_tensor = packet_tensor + self.packet_pos_embedding(positions)
        encoded_packets = self.packet_flow_encoder(packet_tensor, src_key_padding_mask=~packet_valid)

        attn_logits = self.flow_attention(encoded_packets).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~packet_valid, -1e4)
        attn = F.softmax(attn_logits, dim=-1)
        flow_pooled = torch.sum(encoded_packets * attn.unsqueeze(-1), dim=1)

        direct_logits = self.packet_attention(packet_tensor).squeeze(-1)
        direct_logits = direct_logits.masked_fill(~packet_valid, -1e4)
        direct_attn = F.softmax(direct_logits, dim=-1)
        direct_pooled = torch.sum(packet_tensor * direct_attn.unsqueeze(-1), dim=1)
        pooled = flow_pooled + direct_pooled

        pooled = torch.tanh(self.output_layer_1(pooled))
        logits = self.output_layer_2(pooled)

        if tgt is not None:
            loss = classification_loss(logits, tgt, self.class_weights, self.focal_loss_gamma)
            return loss, logits
        return None, logits


def read_label_ids(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return labels_set


def count_labels_num(*paths):
    labels_set = set()
    for path in paths:
        if path is not None:
            labels_set.update(read_label_ids(path))
    max_label = max(labels_set) if labels_set else -1
    if max_label >= 0:
        return max_label + 1
    return 0


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        pretrained_state = torch.load(args.pretrained_model_path, map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0', 'cuda:3': 'cuda:0'})
        model_state = model.state_dict()
        compatible_state = {}
        skipped = []
        for name, tensor in pretrained_state.items():
            if name in model_state and model_state[name].shape == tensor.shape:
                compatible_state[name] = tensor
            else:
                skipped.append(name)
        model.load_state_dict(compatible_state, strict=False)
        if skipped:
            print("Skipped incompatible pretrained tensors: {}".format(", ".join(skipped[:8]) + (" ..." if len(skipped) > 8 else "")))
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    def is_new_module(name):
        name = name[7:] if name.startswith("module.") else name
        return not (name.startswith("embedding.") or name.startswith("encoder."))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    if args.new_module_lr_mult != 1.0:
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in param_optimizer if not is_new_module(n) and not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01,
                'lr': args.learning_rate,
            },
            {
                'params': [p for n, p in param_optimizer if not is_new_module(n) and any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0,
                'lr': args.learning_rate,
            },
            {
                'params': [p for n, p in param_optimizer if is_new_module(n) and not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01,
                'lr': args.learning_rate * args.new_module_lr_mult,
            },
            {
                'params': [p for n, p in param_optimizer if is_new_module(n) and any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0,
                'lr': args.learning_rate * args.new_module_lr_mult,
            },
        ]
    else:
        optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    optimizer_grouped_parameters = [group for group in optimizer_grouped_parameters if group["params"]]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup, args.train_steps)
    return optimizer, scheduler


def set_pretrained_backbone_trainable(model, trainable):
    base_model = model.module if hasattr(model, "module") else model
    for name, param in base_model.named_parameters():
        if name.startswith("embedding.") or name.startswith("encoder."):
            param.requires_grad = trainable


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None, time_delta=None, pkt_len=None, packet_start_mask=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        time_delta_batch = None
        pkt_len_batch = None
        packet_start_batch = None
        if time_delta is not None and pkt_len is not None:
            time_delta_batch = time_delta[i * batch_size: (i + 1) * batch_size, :]
            pkt_len_batch = pkt_len[i * batch_size: (i + 1) * batch_size, :]
        if packet_start_mask is not None:
            packet_start_batch = packet_start_mask[i * batch_size: (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size: (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch, time_delta_batch, pkt_len_batch, packet_start_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None, time_delta_batch, pkt_len_batch, packet_start_batch
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size:, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size:]
        seg_batch = seg[instances_num // batch_size * batch_size:, :]
        time_delta_batch = None
        pkt_len_batch = None
        packet_start_batch = None
        if time_delta is not None and pkt_len is not None:
            time_delta_batch = time_delta[instances_num // batch_size * batch_size:, :]
            pkt_len_batch = pkt_len[instances_num // batch_size * batch_size:, :]
        if packet_start_mask is not None:
            packet_start_batch = packet_start_mask[instances_num // batch_size * batch_size:, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size:, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch, time_delta_batch, pkt_len_batch, packet_start_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None, time_delta_batch, pkt_len_batch, packet_start_batch


def parse_numeric_sequence(raw_values, expected_length):
    if raw_values is None:
        values = []
    else:
        raw_values = raw_values.strip().replace(",", " ")
        values = [float(v) for v in raw_values.split() if len(v) > 0]

    if len(values) == expected_length - 1:
        # Align common case: features are packet-level and do not include CLS.
        values = [0.0] + values
    if len(values) > expected_length:
        values = values[: expected_length]
    if len(values) < expected_length:
        values += [0.0] * (expected_length - len(values))
    return values


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            if "text_b" not in columns:
                text_a = line[columns["text_a"]]
                raw_tokens = [CLS_TOKEN] + args.tokenizer.tokenize(text_a)
                src = args.tokenizer.convert_tokens_to_ids(raw_tokens)
                seg = [1] * len(src)
            else:
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                raw_tokens_a = [CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN]
                src_a = args.tokenizer.convert_tokens_to_ids(raw_tokens_a)
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)
                raw_tokens = raw_tokens_a + args.tokenizer.tokenize(text_b) + [SEP_TOKEN]

            original_len = len(src)
            if args.ignore_packet_start:
                packet_start_mask = [0] * original_len
            elif "packet_start" in columns:
                packet_start_raw = line[columns["packet_start"]]
                packet_start_mask = [1 if value > 0 else 0 for value in parse_numeric_sequence(packet_start_raw, original_len)]
            else:
                packet_start_mask = [1 if token == "[PACKET]" else 0 for token in raw_tokens]

            time_delta = None
            pkt_len = None
            if args.use_aux_features:
                time_raw = line[columns[args.time_feature_column]] if args.time_feature_column in columns else None
                len_raw = line[columns[args.length_feature_column]] if args.length_feature_column in columns else None
                time_delta = parse_numeric_sequence(time_raw, original_len)
                pkt_len = parse_numeric_sequence(len_raw, original_len)
                if args.time_log_scale:
                    time_delta = [np.log1p(max(v, 0.0)) for v in time_delta]
                if args.length_log_scale:
                    pkt_len = [np.log1p(max(v, 0.0)) for v in pkt_len]
                time_delta = [min(v / args.time_feature_scale, 1.0) for v in time_delta]
                pkt_len = [min(v / args.length_feature_scale, 1.0) for v in pkt_len]

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
                packet_start_mask = packet_start_mask[: args.seq_length]
                if args.use_aux_features:
                    time_delta = time_delta[: args.seq_length]
                    pkt_len = pkt_len[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
                packet_start_mask.append(0)
                if args.use_aux_features:
                    time_delta.append(0.0)
                    pkt_len.append(0.0)
            if args.soft_targets and "logits" in columns.keys():
                if args.use_aux_features:
                    dataset.append((src, tgt, seg, soft_tgt, time_delta, pkt_len, packet_start_mask))
                else:
                    dataset.append((src, tgt, seg, soft_tgt, packet_start_mask))
            else:
                if args.use_aux_features:
                    dataset.append((src, tgt, seg, time_delta, pkt_len, packet_start_mask))
                else:
                    dataset.append((src, tgt, seg, packet_start_mask))

    return dataset


def dataset_to_tensors(args, dataset):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    if args.soft_targets:
        soft_tgt = torch.FloatTensor([sample[3] for sample in dataset])
    else:
        soft_tgt = None

    if args.use_aux_features:
        feature_offset = 4 if args.soft_targets else 3
        time_delta = torch.FloatTensor([sample[feature_offset] for sample in dataset])
        pkt_len = torch.FloatTensor([sample[feature_offset + 1] for sample in dataset])
        packet_start_offset = feature_offset + 2
    else:
        time_delta = None
        pkt_len = None
        packet_start_offset = 4 if args.soft_targets else 3

    packet_start_mask = torch.LongTensor([sample[packet_start_offset] for sample in dataset])

    return src, tgt, seg, soft_tgt, time_delta, pkt_len, packet_start_mask


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None,
                time_delta_batch=None, pkt_len_batch=None, packet_start_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)
    if time_delta_batch is not None and pkt_len_batch is not None:
        time_delta_batch = time_delta_batch.to(args.device)
        pkt_len_batch = pkt_len_batch.to(args.device)
    if packet_start_batch is not None:
        packet_start_batch = packet_start_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch, time_delta_batch, pkt_len_batch, packet_start_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset, print_confusion_matrix=False):
    src, tgt, seg, _, time_delta, pkt_len, packet_start_mask = dataset_to_tensors(args, dataset)

    batch_size = args.batch_size

    correct = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    args.model.eval()

    for _, (src_batch, tgt_batch, seg_batch, _, time_delta_batch, pkt_len_batch, packet_start_batch) in enumerate(
        batch_loader(batch_size, src, tgt, seg, None, time_delta, pkt_len, packet_start_mask)
    ):
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        if time_delta_batch is not None and pkt_len_batch is not None:
            time_delta_batch = time_delta_batch.to(args.device)
            pkt_len_batch = pkt_len_batch.to(args.device)
        if packet_start_batch is not None:
            packet_start_batch = packet_start_batch.to(args.device)
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch, None, time_delta_batch, pkt_len_batch, packet_start_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

    if print_confusion_matrix:
        print("Confusion matrix:")
        print(confusion)
        print("Report precision, recall, and f1:")
        eps = 1e-9
        active_label_ids = [
            i for i in range(confusion.size()[0])
            if confusion[:, i].sum().item() > 0
        ]
        for i in active_label_ids:
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            f1 = 0 if (p + r) == 0 else 2 * p * r / (p + r)
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    return correct / len(dataset), confusion


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="space",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space.")

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")
    parser.add_argument("--use_aux_features", action="store_true",
                        help="Use extra timestamp-delta and packet-length sequences.")
    parser.add_argument("--time_feature_column", type=str, default="delta_ts",
                        help="Column name for timestamp delta sequence.")
    parser.add_argument("--length_feature_column", type=str, default="pkt_len",
                        help="Column name for packet length sequence.")
    parser.add_argument("--aux_feature_fusion", choices=["add", "gate", "cross_attention"], default="gate",
                        help="How to fuse auxiliary features with token embeddings.")
    parser.add_argument("--hierarchical_pooling", choices=["none", "packet_mean", "packet_attention"], default="none",
                        help="Optional packet-aware pooling strategy for hierarchical comparison experiments.")
    parser.add_argument("--time_feature_scale", type=float, default=10.0,
                        help="Scale value for timestamp-delta normalization.")
    parser.add_argument("--length_feature_scale", type=float, default=2000.0,
                        help="Scale value for packet-length normalization.")
    parser.add_argument("--time_log_scale", action="store_true",
                        help="Apply log1p before timestamp-delta normalization.")
    parser.add_argument("--length_log_scale", action="store_true",
                        help="Apply log1p before packet-length normalization.")
    parser.add_argument("--ignore_packet_start", action="store_true",
                        help="Ignore packet_start column and treat the sequence as unsegmented.")
    parser.add_argument("--classifier_arch", choices=["etbert", "packet_flow_transformer", "packet_context_flow_transformer", "flowsem_mae", "hybrid_packet_flow_transformer"], default="etbert",
                        help="Classifier backbone architecture.")
    parser.add_argument("--packet_token_layers", type=int, default=1,
                        help="Number of lightweight token-context layers before packet pooling.")
    parser.add_argument("--packet_flow_layers", type=int, default=4,
                        help="Number of packet-flow transformer layers for packet-flow architectures.")
    parser.add_argument("--flowsem_field_slots", type=int, default=32,
                        help="Maximum token/field slots retained per packet for FlowSem-MAE-style tabular modeling.")
    parser.add_argument("--flowsem_direct_token_pooling", action="store_true",
                        help="Add a direct token-attention residual branch to FlowSem-MAE classifiers.")
    parser.add_argument("--flowsem_use_context_encoder", action="store_true",
                        help="Run FlowSem on pretrained encoder-contextualized token states instead of raw embeddings.")
    parser.add_argument("--flowsem_cls_residual", action="store_true",
                        help="Fuse the pretrained encoder CLS state into the FlowSem pooled representation.")
    parser.add_argument("--flowsem_fusion_norm", action="store_true",
                        help="Apply layer normalization after combining FlowSem, direct-token, stats, and CLS branches.")
    parser.add_argument("--flowsem_branch_fusion", choices=["sum", "gate", "concat"], default="sum",
                        help="How FlowSem combines packet, direct-token, flow-stat, and CLS branches.")
    parser.add_argument("--flowsem_no_dual_axis_transformer", action="store_true",
                        help="Ablation: bypass field-axis and temporal-axis TransformerEncoder layers in FlowSem-MAE.")
    parser.add_argument("--flowsem_no_field_axis_transformer", action="store_true",
                        help="Ablation: bypass only the field-axis TransformerEncoder in FlowSem-MAE.")
    parser.add_argument("--flowsem_no_temporal_transformer", action="store_true",
                        help="Ablation: bypass only the temporal-axis TransformerEncoder in FlowSem-MAE.")
    parser.add_argument("--new_module_lr_mult", type=float, default=1.0,
                        help="Learning-rate multiplier for non-embedding/non-encoder modules.")
    parser.add_argument("--freeze_pretrained_epochs", type=int, default=0,
                        help="Freeze pretrained embedding/encoder for the first N fine-tuning epochs.")
    parser.add_argument("--max_train_instances", type=int, default=0,
                        help="Optional cap on training instances for smoke experiments.")
    parser.add_argument("--class_weighting", choices=["none", "legacy_5class", "balanced", "sqrt_balanced"], default="legacy_5class",
                        help="Loss weighting strategy. legacy_5class preserves previous experiments.")
    parser.add_argument("--focal_loss_gamma", type=float, default=0.0,
                        help="Use focal loss when > 0 to emphasize hard weak-class examples.")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config path. Values are expanded to the original ET-BERT/HPTF arguments.")
    args = parser.parse_args(_expand_config_argv())

    args = load_hyperparam(args)
    set_seed(args.seed)

    args.labels_num = count_labels_num(args.train_path, args.dev_path, args.test_path)
    args.class_weights = build_class_weights(args)
    print("Class weighting: {}, weights: {}, focal_gamma: {}".format(
        args.class_weighting,
        args.class_weights,
        args.focal_loss_gamma,
    ))
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    if args.classifier_arch == "packet_flow_transformer":
        model = PacketFlowTransformerClassifier(args)
    elif args.classifier_arch == "packet_context_flow_transformer":
        model = PacketContextFlowTransformerClassifier(args)
    elif args.classifier_arch == "flowsem_mae":
        model = FlowSemMAEClassifier(args)
    elif args.classifier_arch == "hybrid_packet_flow_transformer":
        model = HybridPacketFlowTransformerClassifier(args)
    else:
        model = Classifier(args)
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    if args.max_train_instances > 0:
        trainset = trainset[: args.max_train_instances]
    instances_num = len(trainset)
    batch_size = args.batch_size

    src, tgt, seg, soft_tgt, time_delta, pkt_len, packet_start_mask = dataset_to_tensors(args, trainset)

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    args.model = model

    total_loss, best_result = 0.0, 0.0

    print("Start training.")

    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        if args.freeze_pretrained_epochs > 0:
            trainable = epoch > args.freeze_pretrained_epochs
            set_pretrained_backbone_trainable(model, trainable)
            print("Epoch id: {}, pretrained embedding/encoder trainable: {}".format(epoch, trainable))
        model.train()
        for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch, time_delta_batch, pkt_len_batch, packet_start_batch) in enumerate(
            batch_loader(batch_size, src, tgt, seg, soft_tgt, time_delta, pkt_len, packet_start_mask)
        ):
            loss = train_model(
                args,
                model,
                optimizer,
                scheduler,
                src_batch,
                tgt_batch,
                seg_batch,
                soft_tgt_batch,
                time_delta_batch,
                pkt_len_batch,
                packet_start_batch
            )
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        result = evaluate(args, read_dataset(args, args.dev_path))
        if result[0] > best_result:
            best_result = result[0]
            save_model(model, args.output_model_path)

    if args.test_path is not None:
        print("Test set evaluation.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(args.output_model_path))
        else:
            model.load_state_dict(torch.load(args.output_model_path))
        evaluate(args, read_dataset(args, args.test_path), True)


if __name__ == "__main__":
    main()
