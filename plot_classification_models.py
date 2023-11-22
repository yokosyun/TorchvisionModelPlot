import argparse
import statistics
import os
import yaml
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch import nn
import torchvision


def get_latency(
    model: nn.Module,
    input: Tensor,
    iterations: int = 10,
) -> float:
    latencies = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(input)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    return statistics.median(latencies) / 1.0e3


def plot_models(
    models_info: Dict[str, Any],
    key_x: str,
    key_y: str = "acc1",
    save_dir: str = "results",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots()
    for model_name, model_info in models_info.items():
        if len(model_info[key_x]) > 0:
            plt.scatter(
                model_info[key_x],
                model_info[key_y],
                marker=model_info["marker"],
                label=model_name,
                color=model_info["color"],
            )
    plt.ylabel(key_y)
    plt.xlabel(key_x)
    ax.legend()
    fig.savefig(save_dir + "/" + key_y + "_" + key_x + ".jpg")


def get_model_name_prefix(
    cfg: Dict[str, Any],
    model_name: str,
) -> Optional[str]:
    for model_name_prefix in cfg.keys():
        if model_name.startswith(model_name_prefix):
            return model_name_prefix
    return None


def initialize_empty_list(
    cfg: Dict[str, Any],
) -> None:
    for key in cfg["models"].keys():
        cfg["models"][key]["acc1"] = []
        cfg["models"][key]["acc5"] = []
        cfg["models"][key]["latency"] = []
        cfg["models"][key]["num_params"] = []
        cfg["models"][key]["ops"] = []


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml",
        type=str,
        default="configs/plot_classification_models.yaml",
        help="path to yaml",
    )
    return parser.parse_args()


def main():
    args = get_argparse()
    with open(args.yaml, "r") as f:
        cfg = yaml.safe_load(f)

    initialize_empty_list(cfg)

    models_name = torchvision.models.list_models(module=torchvision.models)

    with torch.no_grad():
        for model_name in models_name:
            weights = torchvision.models.get_model_weights(model_name).DEFAULT
            meta = weights.meta
            crop_size = weights.transforms.keywords.get("crop_size", 224)
            img = torch.rand([1, 3, crop_size, crop_size]).cuda()
            acc = meta.get("_metrics").get("ImageNet-1K")
            num_params = meta.get("num_params")
            _ops = meta.get("_ops")
            acc1 = acc.get("acc@1")
            acc5 = acc.get("acc@5")

            if acc1 >= cfg["min_acc1_thresh"] and acc1 <= cfg["max_acc1_thresh"]:
                print(f"evaluating => {model_name}")

                # vit_h_14 needs to load pretrained weights to avoid input size error
                weights = weights if model_name.startswith("vit_h_14") else None
                model = torchvision.models.get_model(model_name, weights=weights)
                model = model.eval().cuda()

                latency = get_latency(model, img)

                model = model.cpu()
                del model

                model_name = get_model_name_prefix(cfg["models"], model_name)
                cfg["models"][model_name]["acc1"].append(acc1)
                cfg["models"][model_name]["acc5"].append(acc5)
                cfg["models"][model_name]["num_params"].append(num_params)
                cfg["models"][model_name]["ops"].append(_ops)
                cfg["models"][model_name]["latency"].append(latency)

    plot_models(cfg["models"], key_x="latency", key_y="acc1", save_dir=cfg["save_dir"])
    plot_models(cfg["models"], key_x="ops", key_y="acc1", save_dir=cfg["save_dir"])
    plot_models(
        cfg["models"], key_x="num_params", key_y="acc1", save_dir=cfg["save_dir"]
    )

    plot_models(cfg["models"], key_x="latency", key_y="acc5", save_dir=cfg["save_dir"])
    plot_models(cfg["models"], key_x="ops", key_y="acc5", save_dir=cfg["save_dir"])
    plot_models(
        cfg["models"], key_x="num_params", key_y="acc5", save_dir=cfg["save_dir"]
    )


if __name__ == "__main__":
    main()
