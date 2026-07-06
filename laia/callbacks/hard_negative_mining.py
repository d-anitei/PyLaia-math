from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import torch.distributed as dist

import laia.common.logging as log
from laia.data import TextImageFromTextTableDataset

_logger = log.get_logger(__name__)


class HardNegativeMining(pl.Callback):
    """Duplicate the top-k hardest training samples for the next epoch."""

    def __init__(
        self,
        ratio: float = 0.1,
        strategy: str = "ratio",
        start_epoch: int = 1,
        output_txt_table: Optional[str] = None,
    ) -> None:
        if ratio <= 0 or ratio > 1:
            raise ValueError("`ratio` must be in (0, 1].")
        if strategy not in {"ratio", "stddev"}:
            raise ValueError("`strategy` must be either 'ratio' or 'stddev'.")
        if start_epoch < 0:
            raise ValueError("`start_epoch` must be >= 0.")
        self.ratio = ratio
        self.strategy = strategy
        self.start_epoch = start_epoch
        self.output_txt_table = output_txt_table
        self._base_tr_txt_table: Optional[Path] = None

    @staticmethod
    def _reduce_losses_per_sample(losses: Dict[str, List[float]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for sample_id, values in losses.items():
            if values:
                out[sample_id] = float(sum(values) / len(values))
        return out

    @staticmethod
    def _merge_distributed_losses(local_losses: Dict[str, List[float]]) -> Dict[str, List[float]]:
        if not dist.is_available() or not dist.is_initialized():
            return local_losses

        gathered: List[Dict[str, List[float]]] = [
            {} for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(gathered, local_losses)

        merged: Dict[str, List[float]] = defaultdict(list)
        for rank_losses in gathered:
            if not rank_losses:
                continue
            for sample_id, values in rank_losses.items():
                merged[sample_id].extend(values)
        return merged

    @staticmethod
    def _load_training_table(path: Path) -> Tuple[List[str], Dict[str, List[str]]]:
        base_lines = path.read_text().splitlines(keepends=True)
        return HardNegativeMining._load_training_table_from_lines(base_lines)

    @staticmethod
    def _load_training_table_from_lines(
        base_lines: List[str],
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        id_to_lines: Dict[str, List[str]] = defaultdict(list)
        for line in base_lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) < 2:
                continue
            sample_id = parts[0]
            id_to_lines[sample_id].append(line)
        return base_lines, id_to_lines

    @staticmethod
    def _deduplicate_by_sample_id(lines: List[str]) -> List[str]:
        """Keep one line per sample id while preserving original order."""
        seen_ids = set()
        deduped: List[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                deduped.append(line)
                continue
            parts = stripped.split(maxsplit=1)
            if len(parts) < 2:
                deduped.append(line)
                continue
            sample_id = parts[0]
            if sample_id in seen_ids:
                continue
            seen_ids.add(sample_id)
            deduped.append(line)
        return deduped

    def _select_hard_samples(
        self, scored_samples: List[Tuple[str, float]]
    ) -> Tuple[List[Tuple[str, float]], Dict[str, float]]:
        if self.strategy == "ratio":
            num_hard = max(1, ceil(self.ratio * len(scored_samples)))
            hardest = sorted(scored_samples, key=lambda x: x[1], reverse=True)[:num_hard]
            return hardest, {"ratio": self.ratio}

        losses = [sample_loss for _, sample_loss in scored_samples]
        mean_loss = sum(losses) / len(losses)
        variance = sum((loss - mean_loss) ** 2 for loss in losses) / len(losses)
        std_loss = variance ** 0.5
        threshold = mean_loss + std_loss
        hardest = [
            (sample_id, sample_loss)
            for sample_id, sample_loss in scored_samples
            if sample_loss > threshold
        ]
        hardest.sort(key=lambda x: x[1], reverse=True)
        return hardest, {
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "threshold": threshold,
        }

    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        self._base_tr_txt_table = Path(trainer.datamodule.tr_txt_table)

    def on_train_epoch_end(self, trainer, pl_module, *args, **kwargs):
        super().on_train_epoch_end(trainer, pl_module, *args, **kwargs)

        if trainer.current_epoch < self.start_epoch:
            pl_module.monitor_loss = defaultdict(list)
            return

        local_losses = dict(getattr(pl_module, "monitor_loss", {}))
        merged_losses = self._merge_distributed_losses(local_losses)
        reduced_losses = self._reduce_losses_per_sample(merged_losses)
        if not reduced_losses:
            pl_module.monitor_loss = defaultdict(list)
            return

        if self._base_tr_txt_table is None:
            self._base_tr_txt_table = Path(trainer.datamodule.tr_txt_table)

        base_lines, _ = self._load_training_table(self._base_tr_txt_table)
        base_lines = self._deduplicate_by_sample_id(base_lines)
        _, id_to_lines = self._load_training_table_from_lines(base_lines)
        if not id_to_lines:
            pl_module.monitor_loss = defaultdict(list)
            return

        scored_samples = [
            (sample_id, sample_loss)
            for sample_id, sample_loss in reduced_losses.items()
            if sample_id in id_to_lines
        ]
        if not scored_samples:
            pl_module.monitor_loss = defaultdict(list)
            return

        hardest, strategy_stats = self._select_hard_samples(scored_samples)

        hard_lines: List[str] = []
        for sample_id, _ in hardest:
            hard_lines.extend(id_to_lines[sample_id])

        output_path = (
            Path(self.output_txt_table)
            if self.output_txt_table
            else self._base_tr_txt_table.with_name(
                f"{self._base_tr_txt_table.stem}.hardneg.txt"
            )
        )
        if trainer.is_global_zero:
            output_path.write_text("".join(base_lines + hard_lines))
            if self.strategy == "ratio":
                _logger.info(
                    "Hard-negative mining epoch %d: duplicated %d/%d samples (strategy=%s, ratio=%.2f) into %s",
                    trainer.current_epoch,
                    len(hardest),
                    len(scored_samples),
                    self.strategy,
                    strategy_stats["ratio"],
                    output_path,
                )
            else:
                _logger.info(
                    "Hard-negative mining epoch %d: duplicated %d/%d samples (strategy=%s, mean=%.6f, std=%.6f, threshold=%.6f) into %s",
                    trainer.current_epoch,
                    len(hardest),
                    len(scored_samples),
                    self.strategy,
                    strategy_stats["mean_loss"],
                    strategy_stats["std_loss"],
                    strategy_stats["threshold"],
                    output_path,
                )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        tr_img_transform, txt_transform = trainer.datamodule.train_transforms
        trainer.datamodule.tr_txt_table = str(output_path)
        trainer.datamodule.tr_ds = TextImageFromTextTableDataset(
            str(output_path),
            trainer.datamodule.img_dirs,
            img_transform=tr_img_transform,
            txt_transform=txt_transform,
        )

        pl_module.monitor_loss = defaultdict(list)
