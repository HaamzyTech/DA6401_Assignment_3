"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import argparse
import math
import os
from collections import Counter
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import (
    LearnedPositionalEncoding,
    Transformer,
    make_src_mask,
    make_tgt_mask,
)


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing with padding tokens excluded from the target distribution
    and loss normalization.
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        if vocab_size <= 2:
            raise ValueError("vocab_size must be greater than 2")
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: shape [batch * tgt_len, vocab_size].
            target: shape [batch * tgt_len].

        Returns:
            Scalar token-normalized loss.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        target = target.to(device=logits.device)

        with torch.no_grad():
            smooth_denominator = max(self.vocab_size - 2, 1)
            true_dist = torch.full_like(
                log_probs,
                self.smoothing / smooth_denominator,
            )
            true_dist[:, self.pad_idx] = 0.0
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0.0
            true_dist[target == self.pad_idx] = 0.0

        non_pad = target != self.pad_idx
        denom = non_pad.sum().clamp_min(1)
        loss = -(true_dist * log_probs).sum()
        return loss / denom


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation over batches of (src, tgt).
    """
    if is_train and optimizer is None:
        raise ValueError("optimizer is required when is_train=True")

    model.train(is_train)
    pad_idx = getattr(loss_fn, "pad_idx", getattr(model, "pad_idx", 1))

    total_loss = 0.0
    total_tokens = 0

    for src, tgt in data_iter:
        src = src.to(device)
        tgt = tgt.to(device)

        if tgt.size(1) < 2:
            continue

        tgt_input = tgt[:, :-1]
        tgt_gold = tgt[:, 1:]
        src_mask = make_src_mask(src, pad_idx)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx)
        ntokens = int((tgt_gold != pad_idx).sum().item())

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                tgt_gold.reshape(-1),
            )

            if is_train:
                loss.backward()
                global_step = int(getattr(model, "_global_train_step", 0))
                if global_step < 1000:
                    q_norm, k_norm = _attention_projection_grad_norms(model)
                    _wandb_log(
                        {
                            "step": global_step,
                            "epoch": epoch_num,
                            "grad_norm/query": q_norm,
                            "grad_norm/key": k_norm,
                        }
                    )
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                model._global_train_step = global_step + 1

        total_loss += float(loss.item()) * max(ntokens, 1)
        total_tokens += ntokens

    return total_loss / max(total_tokens, 1)


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.
    """
    was_training = model.training
    model.eval()

    src = src.to(device)
    src_mask = src_mask.to(device)
    if src.dim() == 1:
        src = src.unsqueeze(0)
    if src_mask.dim() == 3:
        src_mask = src_mask.unsqueeze(1)

    pad_idx = getattr(model, "pad_idx", 1)

    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx)
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            next_word = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            ys = torch.cat(
                [
                    ys,
                    torch.tensor([[next_word]], dtype=torch.long, device=device),
                ],
                dim=1,
            )
            if next_word == end_symbol:
                break

    if was_training:
        model.train()
    return ys


def _vocab_index(vocab, token: str, default: int) -> int:
    if vocab is None:
        return default
    if hasattr(vocab, "stoi"):
        return vocab.stoi.get(token, default)
    if hasattr(vocab, "get_stoi"):
        return vocab.get_stoi().get(token, default)
    if isinstance(vocab, dict):
        return vocab.get(token, default)
    try:
        return vocab[token]
    except Exception:
        return default


def _lookup_token(vocab, index: int) -> str:
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(index)
    if hasattr(vocab, "itos"):
        return vocab.itos[index]
    if hasattr(vocab, "get_itos"):
        return vocab.get_itos()[index]
    return str(index)


def _indices_to_tokens(
    indices: Iterable[int],
    vocab,
    pad_idx: int,
    sos_idx: int,
    eos_idx: int,
) -> List[str]:
    tokens = []
    for idx in indices:
        idx = int(idx)
        if idx == eos_idx:
            break
        if idx in {pad_idx, sos_idx}:
            continue
        token = _lookup_token(vocab, idx)
        if token not in {"<pad>", "<sos>", "<eos>"}:
            tokens.append(token)
    return tokens


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _corpus_bleu(
    predictions: Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
    max_n: int = 4,
) -> float:
    pred_len = sum(len(pred) for pred in predictions)
    ref_len = sum(len(ref) for ref in references)
    if pred_len == 0:
        return 0.0

    log_precisions = []
    for n in range(1, max_n + 1):
        matches = 0
        possible = 0
        for pred, ref in zip(predictions, references):
            pred_ngrams = _ngrams(pred, n)
            ref_ngrams = _ngrams(ref, n)
            possible += max(len(pred) - n + 1, 0)
            overlap = pred_ngrams & ref_ngrams
            matches += sum(overlap.values())

        if possible == 0:
            continue
        precision = matches / possible if matches > 0 else 1.0 / (2.0 * possible)
        log_precisions.append(math.log(precision))

    if not log_precisions:
        return 0.0

    geo_mean = math.exp(sum(log_precisions) / len(log_precisions))
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / pred_len)
    return 100.0 * brevity_penalty * geo_mean


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with a corpus-level BLEU score in [0, 100].
    """
    pad_idx = _vocab_index(tgt_vocab, "<pad>", getattr(model, "pad_idx", 1))
    sos_idx = _vocab_index(tgt_vocab, "<sos>", getattr(model, "sos_idx", 2))
    eos_idx = _vocab_index(tgt_vocab, "<eos>", getattr(model, "eos_idx", 3))

    predictions = []
    references = []

    was_training = model.training
    model.eval()

    for src_batch, tgt_batch in test_dataloader:
        if src_batch.dim() == 1:
            src_batch = src_batch.unsqueeze(0)
        if tgt_batch.dim() == 1:
            tgt_batch = tgt_batch.unsqueeze(0)

        for i in range(src_batch.size(0)):
            src = src_batch[i : i + 1].to(device)
            src_mask = make_src_mask(src, pad_idx)
            pred_ids = greedy_decode(
                model,
                src,
                src_mask,
                max_len=max_len,
                start_symbol=sos_idx,
                end_symbol=eos_idx,
                device=device,
            ).squeeze(0)

            predictions.append(
                _indices_to_tokens(pred_ids.tolist(), tgt_vocab, pad_idx, sos_idx, eos_idx)
            )
            references.append(
                _indices_to_tokens(
                    tgt_batch[i].tolist(),
                    tgt_vocab,
                    pad_idx,
                    sos_idx,
                    eos_idx,
                )
            )

    if was_training:
        model.train()
    return _corpus_bleu(predictions, references)


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model, optimizer, scheduler, epoch, and reconstruction config.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    model_config = getattr(model, "model_config", None)
    if model_config is None:
        model_config = {
            "src_vocab_size": model.src_vocab_size,
            "tgt_vocab_size": model.tgt_vocab_size,
            "d_model": model.d_model,
            "N": model.N,
            "num_heads": model.num_heads,
            "d_ff": model.d_ff,
            "dropout": model.dropout,
        }

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": dict(model_config),
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model and optionally optimizer/scheduler state. Returns saved epoch.
    """
    try:
        map_location = next(model.parameters()).device
    except StopIteration:
        map_location = "cpu"

    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    model._checkpoint_loaded = True

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return int(checkpoint.get("epoch", 0))


def _set_attention_scaling(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "use_scaling"):
            module.use_scaling = enabled


def _attention_projection_grad_norms(model: nn.Module) -> tuple[float, float]:
    q_sq = 0.0
    k_sq = 0.0
    for module in model.modules():
        if hasattr(module, "q_linear") and module.q_linear.weight.grad is not None:
            q_sq += float(module.q_linear.weight.grad.detach().norm(2).item() ** 2)
        if hasattr(module, "k_linear") and module.k_linear.weight.grad is not None:
            k_sq += float(module.k_linear.weight.grad.detach().norm(2).item() ** 2)
    return math.sqrt(q_sq), math.sqrt(k_sq)


def _wandb_log(data: dict) -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.log(data)


def _mean_correct_token_confidence(
    model: Transformer,
    data_loader: DataLoader,
    device: str,
    pad_idx: int,
) -> float:
    was_training = model.training
    model.eval()

    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            if tgt.size(1) < 2:
                continue
            tgt_input = tgt[:, :-1]
            tgt_gold = tgt[:, 1:]
            logits = model(
                src,
                tgt_input,
                make_src_mask(src, pad_idx),
                make_tgt_mask(tgt_input, pad_idx),
            )
            probs = F.softmax(logits, dim=-1)
            gold_probs = probs.gather(-1, tgt_gold.unsqueeze(-1)).squeeze(-1)
            mask = tgt_gold != pad_idx
            confidence = gold_probs[mask].mean().item() if mask.any() else 0.0
            if was_training:
                model.train()
            return float(confidence)

    if was_training:
        model.train()
    return 0.0


def _evaluate_token_accuracy(
    model: Transformer,
    data_loader: DataLoader,
    device: str,
    pad_idx: int,
) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            if tgt.size(1) < 2:
                continue
            tgt_input = tgt[:, :-1]
            tgt_gold = tgt[:, 1:]
            logits = model(
                src,
                tgt_input,
                make_src_mask(src, pad_idx),
                make_tgt_mask(tgt_input, pad_idx),
            )
            pred = logits.argmax(dim=-1)
            mask = tgt_gold != pad_idx
            correct += int(((pred == tgt_gold) & mask).sum().item())
            total += int(mask.sum().item())

    if was_training:
        model.train()
    return correct / max(total, 1)


def _log_attention_heatmaps(
    model: Transformer,
    data_loader: DataLoader,
    src_vocab,
    device: str,
    pad_idx: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for src, _ in data_loader:
            src = src[:1].to(device)
            model.encode(src, make_src_mask(src, pad_idx))
            attn = model.encoder.layers[-1].self_attn.last_attn_weights
            if attn is None:
                break
            attn = attn[0].detach().cpu()
            tokens = [
                src_vocab.lookup_token(int(idx))
                if hasattr(src_vocab, "lookup_token")
                else str(int(idx))
                for idx in src[0].detach().cpu().tolist()
            ]
            images = {}
            for head_idx in range(min(attn.size(0), 8)):
                fig, ax = plt.subplots(figsize=(7, 6))
                ax.imshow(attn[head_idx].numpy(), aspect="auto", cmap="viridis")
                ax.set_title(f"Last Encoder Layer - Head {head_idx}")
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=90)
                ax.set_yticklabels(tokens)
                fig.tight_layout()
                images[f"attention/head_{head_idx}"] = wandb.Image(fig)
                plt.close(fig)
            wandb.log(images)
            break

    if was_training:
        model.train()


def _build_arg_parser(default_config: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DA6401 Assignment 3 Transformer experiments."
    )
    parser.add_argument("--wandb-project", default="DA6401_Assignment_3")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
    )

    parser.add_argument("--batch-size", type=int, default=default_config["batch_size"])
    parser.add_argument("--num-epochs", type=int, default=default_config["num_epochs"])
    parser.add_argument("--d-model", type=int, default=default_config["d_model"])
    parser.add_argument("--num-layers", dest="N", type=int, default=default_config["N"])
    parser.add_argument("--num-heads", type=int, default=default_config["num_heads"])
    parser.add_argument("--d-ff", type=int, default=default_config["d_ff"])
    parser.add_argument("--dropout", type=float, default=default_config["dropout"])

    parser.add_argument("--warmup-steps", type=int, default=default_config["warmup_steps"])
    parser.add_argument("--base-lr", type=float, default=default_config["base_lr"])
    parser.add_argument("--fixed-lr", type=float, default=default_config["fixed_lr"])
    lr_group = parser.add_mutually_exclusive_group()
    lr_group.add_argument("--use-noam", dest="use_noam", action="store_true")
    lr_group.add_argument("--no-noam", dest="use_noam", action="store_false")
    parser.set_defaults(use_noam=default_config["use_noam"])

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=default_config["label_smoothing"],
    )

    pos_group = parser.add_mutually_exclusive_group()
    pos_group.add_argument(
        "--learned-positional-encoding",
        dest="learned_positional_encoding",
        action="store_true",
    )
    pos_group.add_argument(
        "--sinusoidal-positional-encoding",
        dest="learned_positional_encoding",
        action="store_false",
    )
    parser.set_defaults(
        learned_positional_encoding=default_config["learned_positional_encoding"]
    )

    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument(
        "--scale-attention",
        dest="use_scaled_attention",
        action="store_true",
    )
    scale_group.add_argument(
        "--no-scale-attention",
        dest="use_scaled_attention",
        action="store_false",
    )
    parser.set_defaults(use_scaled_attention=default_config["use_scaled_attention"])

    parser.add_argument("--checkpoint-path", default=default_config["checkpoint_path"])
    parser.add_argument("--num-workers", type=int, default=default_config["num_workers"])
    parser.add_argument("--max-decode-len", type=int, default=default_config["max_decode_len"])
    return parser


def run_training_experiment() -> None:
    """
    Set up and run the full Multi30k Transformer experiment.
    """
    try:
        import wandb
    except ImportError:  # pragma: no cover - depends on local environment
        wandb = None

    from dataset import PAD_IDX, Multi30kDataset
    from lr_scheduler import NoamScheduler

    default_config = {
        "batch_size": 64,
        "num_epochs": 10,
        "d_model": 512,
        "N": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "base_lr": 1.0,
        "fixed_lr": 1e-4,
        "use_noam": True,
        "label_smoothing": 0.1,
        "learned_positional_encoding": False,
        "use_scaled_attention": True,
        "checkpoint_path": "checkpoint.pt",
        "num_workers": 0,
        "max_decode_len": 100,
    }

    args = _build_arg_parser(default_config).parse_args()
    config = default_config.copy()
    config.update(vars(args))

    run = None
    if wandb is not None:
        run = wandb.init(
            project=config["wandb_project"],
            name=config["run_name"],
            mode=config["wandb_mode"],
            config=config,
        )
        config = dict(wandb.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_set = Multi30kDataset(split="train")
    val_set = Multi30kDataset(split="validation")
    test_set = Multi30kDataset(split="test")

    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=Multi30kDataset.collate_fn,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=Multi30kDataset.collate_fn,
        num_workers=config["num_workers"],
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        collate_fn=Multi30kDataset.collate_fn,
        num_workers=config["num_workers"],
    )

    model = Transformer(
        src_vocab_size=len(train_set.src_vocab),
        tgt_vocab_size=len(train_set.tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(device)
    model.src_vocab = train_set.src_vocab
    model.tgt_vocab = train_set.tgt_vocab
    model.src_tokenizer = train_set.src_tokenizer

    if config["learned_positional_encoding"]:
        model.src_pos = LearnedPositionalEncoding(config["d_model"], config["dropout"]).to(device)
        model.tgt_pos = LearnedPositionalEncoding(config["d_model"], config["dropout"]).to(device)

    _set_attention_scaling(model, bool(config["use_scaled_attention"]))

    lr = config["base_lr"] if config["use_noam"] else config["fixed_lr"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = (
        NoamScheduler(
            optimizer,
            d_model=config["d_model"],
            warmup_steps=config["warmup_steps"],
        )
        if config["use_noam"]
        else None
    )
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_set.tgt_vocab),
        pad_idx=PAD_IDX,
        smoothing=config["label_smoothing"],
    )

    best_val_loss = float("inf")
    for epoch in range(config["num_epochs"]):
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
        )
        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
        )
        confidence = _mean_correct_token_confidence(model, val_loader, device, PAD_IDX)
        val_accuracy = _evaluate_token_accuracy(model, val_loader, device, PAD_IDX)
        current_lr = optimizer.param_groups[0]["lr"]

        if wandb is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "lr": current_lr,
                    "prediction_confidence": confidence,
                }
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                path=config["checkpoint_path"],
            )

    bleu = evaluate_bleu(
        model,
        test_loader,
        train_set.tgt_vocab,
        device=device,
        max_len=config["max_decode_len"],
    )
    _log_attention_heatmaps(model, val_loader, train_set.src_vocab, device, PAD_IDX)
    if wandb is not None:
        wandb.log({"test_bleu": bleu})
        if run is not None:
            run.finish()
    else:
        print(f"test_bleu={bleu:.4f}")


if __name__ == "__main__":
    run_training_experiment()


