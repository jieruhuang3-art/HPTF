import argparse
import json

import torch
import torch.nn as nn

from hptf.evaluation.metrics import compute_metrics
from hptf.training import finetune


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    ft_parser = argparse.ArgumentParser()
    finetune.finetune_opts(ft_parser)
    ft_args = ft_parser.parse_args(finetune._config_to_argv(args.config))
    ft_args = finetune.load_hyperparam(ft_args)
    ft_args.labels_num = finetune.count_labels_num(ft_args.train_path, ft_args.dev_path, ft_args.test_path)
    ft_args.class_weights = None
    ft_args.tokenizer = finetune.str2tokenizer[ft_args.tokenizer](ft_args)
    if ft_args.classifier_arch == "flowsem_mae":
        model = finetune.FlowSemMAEClassifier(ft_args)
    elif ft_args.classifier_arch == "packet_flow_transformer":
        model = finetune.PacketFlowTransformerClassifier(ft_args)
    elif ft_args.classifier_arch == "packet_context_flow_transformer":
        model = finetune.PacketContextFlowTransformerClassifier(ft_args)
    elif ft_args.classifier_arch == "hybrid_packet_flow_transformer":
        model = finetune.HybridPacketFlowTransformerClassifier(ft_args)
    else:
        model = finetune.Classifier(ft_args)
    ft_args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.checkpoint, map_location=ft_args.device), strict=False)
    model = model.to(ft_args.device)
    model.eval()

    dataset = finetune.read_dataset(ft_args, ft_args.test_path)
    src, tgt, seg, _, time_delta, pkt_len, packet_start_mask = finetune.dataset_to_tensors(ft_args, dataset)
    y_true, y_pred = [], []
    for src_batch, tgt_batch, seg_batch, _, time_batch, len_batch, packet_start_batch in finetune.batch_loader(
        ft_args.batch_size, src, tgt, seg, None, time_delta, pkt_len, packet_start_mask
    ):
        src_batch = src_batch.to(ft_args.device)
        tgt_batch = tgt_batch.to(ft_args.device)
        seg_batch = seg_batch.to(ft_args.device)
        time_batch = time_batch.to(ft_args.device) if time_batch is not None else None
        len_batch = len_batch.to(ft_args.device) if len_batch is not None else None
        packet_start_batch = packet_start_batch.to(ft_args.device) if packet_start_batch is not None else None
        with torch.no_grad():
            _, logits = model(src_batch, tgt_batch, seg_batch, None, time_batch, len_batch, packet_start_batch)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        y_true.extend(tgt_batch.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
