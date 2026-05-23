#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from hptf.uer.encoders import str2encoder
from hptf.uer.layers import str2embedding
from hptf.uer.model_saver import save_model
from hptf.uer.opts import finetune_opts
from hptf.uer.utils import str2optimizer, str2scheduler, str2tokenizer
from hptf.uer.utils.config import load_hyperparam
from hptf.uer.utils.constants import MASK_TOKEN
from hptf.uer.utils.seed import set_seed


def load_classifier_module():
    from hptf.training import finetune as module
    return module


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
    pretraining = cfg.get("pretraining", {})
    training = cfg.get("training", {})
    argv = []
    add("--train_path", data.get("train_path"), argv)
    add("--dev_path", data.get("valid_path", data.get("train_path")), argv)
    add("--test_path", data.get("valid_path"), argv)
    add("--vocab_path", data.get("vocab_path"), argv)
    add("--config_path", model.get("config_path", "configs/bert_base_config.json"), argv)
    add("--seq_length", model.get("seq_length", model.get("max_seq_length", 128)), argv)
    add("--use_aux_features", model.get("use_auxiliary_features", True), argv)
    add("--aux_feature_fusion", model.get("aux_feature_fusion", "gate"), argv)
    add("--mask_prob", pretraining.get("mask_probability"), argv)
    if pretraining.get("lambda_time") == pretraining.get("lambda_length"):
        add("--aux_loss_weight", pretraining.get("lambda_time"), argv)
    else:
        add("--aux_loss_weight", pretraining.get("aux_loss_weight"), argv)
    add("--batch_size", training.get("batch_size"), argv)
    add("--learning_rate", training.get("learning_rate"), argv)
    add("--epochs_num", training.get("epochs", training.get("epochs_num")), argv)
    add("--warmup", training.get("warmup_ratio", training.get("warmup")), argv)
    add("--seed", training.get("seed"), argv)
    output_dir = training.get("output_dir")
    if output_dir:
        add("--output_model_path", os.path.join(output_dir, "pytorch_model.bin"), argv)
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


class TrafficMlmAuxPretrainer(nn.Module):
    def __init__(self, args):
        super(TrafficMlmAuxPretrainer, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.use_aux_features = args.use_aux_features
        self.aux_feature_fusion = args.aux_feature_fusion

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

        self.mlm_head = nn.Linear(args.hidden_size, len(args.tokenizer.vocab))
        self.time_head = nn.Linear(args.hidden_size, 1)
        self.length_head = nn.Linear(args.hidden_size, 1)

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

    def forward(self, src, seg, mask_positions, time_delta, pkt_len):
        emb = self.embedding(src, seg)
        emb = self._fuse_aux_features(emb, seg, time_delta, pkt_len)
        output, _ = self._unpack_encoder_output(self.encoder(emb, seg))

        mlm_logits = self.mlm_head(output)
        time_pred = self.time_head(output).squeeze(-1)
        length_pred = self.length_head(output).squeeze(-1)
        return mlm_logits, time_pred, length_pred


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay_rate": 0.01},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay_rate": 0.0},
    ]
    optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    if args.scheduler == "constant":
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler == "constant_with_warmup":
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps * args.warmup, args.train_steps)
    return optimizer, scheduler


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="space")
    parser.add_argument("--use_aux_features", action="store_true")
    parser.add_argument("--time_feature_column", type=str, default="delta_ts")
    parser.add_argument("--length_feature_column", type=str, default="pkt_len")
    parser.add_argument("--aux_feature_fusion", choices=["add", "gate", "cross_attention"], default="gate")
    parser.add_argument("--time_feature_scale", type=float, default=10.0)
    parser.add_argument("--length_feature_scale", type=float, default=2000.0)
    parser.add_argument("--time_log_scale", action="store_true")
    parser.add_argument("--length_log_scale", action="store_true")
    parser.add_argument("--ignore_packet_start", action="store_true")
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--max_train_instances", type=int, default=0)
    parser.add_argument("--aux_loss_weight", type=float, default=0.2)
    parser.add_argument("--disable_mlm_loss", action="store_true",
                        help="Ablation: remove masked-token prediction from the pretraining objective.")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config path. Values are expanded to the original traffic MLM+aux arguments.")
    args = parser.parse_args(_expand_config_argv())

    args = load_hyperparam(args)
    set_seed(args.seed)
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    args.soft_targets = False

    run_classifier = load_classifier_module()
    trainset = run_classifier.read_dataset(args, args.train_path)
    random.shuffle(trainset)
    if args.max_train_instances > 0:
        trainset = trainset[: args.max_train_instances]
    src, _, seg, _, time_delta, pkt_len, _ = run_classifier.dataset_to_tensors(args, trainset)

    model = TrafficMlmAuxPretrainer(args)
    if args.pretrained_model_path is not None:
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location={"cuda:1": "cuda:0", "cuda:2": "cuda:0", "cuda:3": "cuda:0"}), strict=False)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)

    mask_id = args.tokenizer.convert_tokens_to_ids([MASK_TOKEN])[0]
    batch_size = args.batch_size
    args.train_steps = int(len(trainset) * args.epochs_num / batch_size) + 1
    optimizer, scheduler = build_optimizer(args, model)

    print("Pretraining instances:", len(trainset))
    print("Batch size:", batch_size)
    print("Start traffic MLM+aux pretraining.")
    if args.disable_mlm_loss:
        print("Ablation objective: disabled MLM loss; optimizing aux time/length losses only.")

    total_loss = 0.0
    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        model.train()
        for i, (src_batch, _, seg_batch, _, time_batch, len_batch, _) in enumerate(
            run_classifier.batch_loader(batch_size, src, torch.zeros(src.size(0), dtype=torch.long), seg, None, time_delta, pkt_len, None)
        ):
            src_batch = src_batch.to(args.device)
            seg_batch = seg_batch.to(args.device)
            time_batch = time_batch.to(args.device)
            len_batch = len_batch.to(args.device)

            mask_positions = (torch.rand(src_batch.size(), device=args.device) < args.mask_prob) & (seg_batch > 0) & (src_batch != 0)
            if not mask_positions.any():
                mask_positions[:, 0] = True
            labels = src_batch.clone()
            masked_src = src_batch.clone()
            masked_src[mask_positions] = mask_id

            mlm_logits, time_pred, len_pred = model(masked_src, seg_batch, mask_positions, time_batch, len_batch)
            mlm_loss = F.cross_entropy(mlm_logits[mask_positions], labels[mask_positions])
            time_loss = F.mse_loss(time_pred[mask_positions], time_batch[mask_positions])
            len_loss = F.mse_loss(len_pred[mask_positions], len_batch[mask_positions])
            if args.disable_mlm_loss:
                loss = args.aux_loss_weight * (time_loss + len_loss)
            else:
                loss = mlm_loss + args.aux_loss_weight * (time_loss + len_loss)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Pretrain steps: {}, Avg loss: {:.4f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

    save_model(model, args.output_model_path)
    print("saved traffic pretraining model to", args.output_model_path)


if __name__ == "__main__":
    main()
