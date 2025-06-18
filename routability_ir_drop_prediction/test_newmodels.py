# Copyright 2022 CircuitNet. All rights reserved.

"""Optimised test script that fully utilises the GPU.

* Uses a `DataLoader` **only if** `build_dataset` does *not* already return one – avoids the double‑wrapping error you just saw.
* Keeps all the other GPU‑optimisation tweaks (batching, pin‑memory, AMP, no‑grad, async H2D, etc.).

Drop‑in replacement for the original `test.py`.
"""

from __future__ import print_function

import os
import os.path as osp
import glob
import json
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm import tqdm

from datasets.build_dataset import build_dataset
from utils.metrics import build_metric, build_roc_prc_metric
from models.build_model import build_model
from utils.configs import Parser
import csv


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


def test(suffix: str, iter):
    # ------------------------------------------------------------------
    # 1. Parse args – extra options for GPU performance
    # ------------------------------------------------------------------
    argp = Parser()
    arg = argp.parser.parse_args()
    arg_dict = vars(arg)

    if arg.arg_file is not None:
        with open(arg.arg_file, "rt") as f:
            arg_dict.update(json.load(f))

    arg_dict["ann_file"] = arg_dict["ann_file_test"]
    arg_dict["test_mode"] = True

    arg_dict['save_path'] = os.path.join('work_dir', f"congestion_gpdl{suffix}")
    if os.path.exists(os.path.join(arg_dict['save_path'], f"model_iters_{iter}.pth")):
        pretrained_path = os.path.join(arg_dict['save_path'], f"model_iters_{iter}.pth")
    else:
        return
        # find all .pth files in save_dir
        pth_files = glob.glob(os.path.join(arg_dict['save_path'], "*.pth"))
        # pick the one with the latest modification time
        pretrained_path = max(pth_files, key=os.path.getmtime)
    arg_dict['pretrained'] = pretrained_path
    print("Testing model at", pretrained_path)
    if suffix in ['_deep_in_leaky', '_deep', '_squeeze_in_leaky', '_squeeze', '_doubleUNet']:
        arg_dict['model_type'] += suffix


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
    model = build_model(arg_dict).to(device).eval()
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
    # ... inside your test function, replacing the placeholder code:

    output_csv = os.path.join(arg_dict["save_path"], "results.csv")
    batches = len(loader)
    file_exists = os.path.isfile(output_csv)

    with open(output_csv, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["iterations", "NRMS", "SSIM", "EMD"])
        writer.writerow([
            iter,
            metric_totals["NRMS"] / batches,
            metric_totals["SSIM"] / batches,
            metric_totals["EMD"] / batches,
        ])
    # output_path = os.path.join(arg_dict["save_path"], "result.txt")
    # with open(output_path, "w") as f:
    #     batches = len(loader)
    #     for name, total in metric_totals.items():
    #         f.write(f"===> Avg. {name}: {total / batches:.4f}\n")

    #     # ------------------------------------------------------------------
    #     # 7. Optional ROC/PRC
    #     # ------------------------------------------------------------------
    #     if arg_dict.get("plot_roc", False):
    #         roc_metric, _ = build_roc_prc_metric(**arg_dict)
    #         f.write(f"\n===> AUC of ROC. {roc_metric:.4f}\n")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    for iter in range(10000, 200001, 10000):
    # ['_pretrained', '_sft', '', '_bn_act', '_deep', '_squeeze', '_doubleUNet']
        # run three suffix tests in parallel
        suffixes = ['']
        processes = []
        for suffix in suffixes:
            p = mp.Process(target=test, args=(suffix, iter))
            p.start()
            processes.append(p)
        if iter % 50000 == 0:
            for p in processes:
                p.join()
