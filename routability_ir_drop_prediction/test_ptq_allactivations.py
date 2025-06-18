# Copyright 2022 CircuitNet. All rights reserved.

"""Optimised test script that fully utilises the GPU.

* Uses a `DataLoader` **only if** `build_dataset` does *not* already return one – avoids the double‑wrapping error you just saw.
* Keeps all the other GPU‑optimisation tweaks (batching, pin‑memory, AMP, no‑grad, async H2D, etc.).

Drop‑in replacement for the original `test.py`.
"""

from __future__ import print_function

import os
import os.path as osp
import json
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.build_dataset import build_dataset
from utils.metrics import build_metric, build_roc_prc_metric
from models.build_model import build_model
from utils.configs import Parser
import csv
import argparse
import bitsandbytes as bnb
import sys

def _maybe_dataloader(dataset: Any, arg_dict):
    """Return a DataLoader unless *dataset* is already one."""
    if isinstance(dataset, DataLoader):
        # Already batched – we respect the user's loader as‑is.
        return dataset

    return DataLoader(
        dataset,
        batch_size=arg_dict.get("batch_size", 256),
        shuffle=False,
        num_workers=arg_dict.get("num_workers", 16),
        pin_memory=arg_dict.get("pin_memory", True),
        drop_last=False,
        prefetch_factor=4,
    )


def _save_predictions(preds_cpu, paths, save_root):
    os.makedirs(save_root, exist_ok=True)
    for base, pred in zip(paths, preds_cpu):
        np.save(osp.join(save_root, f"{osp.splitext(osp.basename(base))[0]}.npy"), pred)


def test(quant_bit: int = 8):
    # ------------------------------------------------------------------
    # 1. Parse args – extra options for GPU performance
    # ------------------------------------------------------------------
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)
    arg_dict['pretrained'] = f'./work_dir/congestion_gpdl/model_iters_200000.pth'
    arg_dict['quant_bit'] = quant_bit
    arg_dict['quant_part'] = []

    for train_mode in ['encoder_activations', 'decoder_activations', 'all_activations']:
        arg_dict['train_mode'] = train_mode

        if arg.arg_file is not None:
            with open(arg.arg_file, "rt") as f:
                arg_dict.update(json.load(f))

        arg_dict["ann_file"] = arg_dict["ann_file_test"]
        arg_dict["test_mode"] = True

        # ------------------------------------------------------------------
        # 2. Dataset / dataloader
        # ------------------------------------------------------------------
        print("===> Loading dataset …")
        dataset = build_dataset(arg_dict)
        loader = _maybe_dataloader(dataset, arg_dict)

        # ------------------------------------------------------------------
        # 3. Model
        # ------------------------------------------------------------------
        device = torch.device("cpu" if arg_dict.get("cpu", False) else "cuda")
        # 3a. Build and post-quantize the model to 8-bit (dynamic quantization)
        model_fp32 = build_model(arg_dict)
        model = model_fp32.to(device).eval()

        # 3b. Set up AMP/autocast context
        autocast_ctx = (
            torch.cuda.amp.autocast
            if (device.type == "cuda" and arg_dict.get("amp", False))
            else nullcontext
        )

        # ------------------------------------------------------------------
        # 4. Metrics
        # ------------------------------------------------------------------
        metrics = {k: build_metric(k) for k in arg_dict["eval_metric"]}
        metric_totals = {k: 0.0 for k in arg_dict["eval_metric"]}

        # ------------------------------------------------------------------
        # 5. Inference loop
        # ------------------------------------------------------------------
        print("===> Running inference …")
        with torch.no_grad(), tqdm(total=len(loader), dynamic_ncols=True) as bar:
            for features, labels, paths in loader:
                features = features.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast_ctx():
                    preds = model(features).squeeze(1)

                preds_cpu = preds.detach().cpu()
                labels_cpu = labels.detach().cpu()

                for name, fn in metrics.items():
                    metric_totals[name] += fn(labels_cpu, preds_cpu)

                if arg_dict.get("plot_roc", False):
                    _save_predictions(
                        preds_cpu.numpy(),
                        paths,
                        osp.join(arg_dict["save_path"], "test_result"),
                    )

                bar.update(1)

        # ------------------------------------------------------------------
        # 6. Print averaged metrics
        # ------------------------------------------------------------------
        output_prefix = train_mode.partition('_')[0]
        output_path = os.path.join("./work_dir/ptq_activations", f"{output_prefix}_{arg_dict['quant_bit']}bits.txt")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            batches = len(loader)
            for name, total in metric_totals.items():
                f.write(f"===> Avg. {name}: {total / batches:.4f}\n")

            # ------------------------------------------------------------------
            # 7. Optional ROC/PRC
            # ------------------------------------------------------------------
            if arg_dict.get("plot_roc", False):
                roc_metric, _ = build_roc_prc_metric(**arg_dict)
                f.write(f"\n===> AUC of ROC. {roc_metric:.4f}\n")

if __name__ == "__main__":
    for quant_bit in [4, 8, 16, 32]:
        test(quant_bit=quant_bit)