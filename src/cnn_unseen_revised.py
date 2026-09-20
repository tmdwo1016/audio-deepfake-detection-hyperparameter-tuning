"""Archived strict-search CNN implementation and reusable fitting primitives.

The user stopped the 36-trial strict search before any trial finished. Its
search and Test entry points are disabled. ``src.cnn_unseen_fixed`` reuses the
verified input, DataLoader and one-trial fitting primitives for the new
fixed-hyperparameter transfer protocol.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.cnn_modeling import LogMelCNN, LogMelDataset, predict_scores
from src.modeling_evaluation import (
    assert_group_disjoint,
    evaluate_binary_predictions,
    select_best_candidate,
)

HOLDOUTS = ("musicgen", "udio")
PROTOCOL_SHA256 = "95f1546ce8e3fcedb66e5ef52488c9255e414965514b5163e1a62b9052e4941e"
SEED = 42
BASELINE = dict(
    lr=1e-3,
    dropout=0.3,
    weight_decay=1e-4,
    batch_size=32,
    max_epochs=20,
    patience=4,
    optimizer="AdamW",
    architecture="historical_4_conv_blocks",
)
SEARCH = dict(
    lr=(1e-4, 3e-4, 1e-3),
    dropout=(0.2, 0.3, 0.5),
    batch_size=(16, 32, 64),
    weight_decay=(0.0, 1e-4, 1e-3),
    max_epochs=30,
    patience=4,
)
MEL_SETTINGS = dict(
    sr=24000,
    segment_sec=10,
    n_fft=1024,
    hop_length=240,
    n_mels=128,
    fmax=12000,
    center=True,
    power=2.0,
    top_db=80,
    ref="max",
    fixed_scale="(dB+40)/40",
)


def sha256_file(path: Path) -> str:
    """Hash an input file so a checkpoint can reject changed manifests."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def id_hash(frame: pd.DataFrame) -> str:
    """Hash the ordered included segment IDs of one Train or Validation mask."""
    return hashlib.sha256("\n".join(frame.segment_id.astype(str)).encode()).hexdigest()


def seed_everything(seed: int = SEED) -> None:
    """Reset each candidate to an independent common Python/NumPy/Torch seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class UnseenCNNExperiment:
    """Own isolated holdout search, freeze, and final Test artifact paths."""

    def __init__(self, root: Path, run_id: str):
        self.root = Path(root)
        self.run_id = run_id
        self.manifest_path = self.root / "data/metadata/segment_manifest_10s.csv"
        self.index_path = self.root / "data/processed/logmel/logmel_10s_index.csv"
        self.cache_path = self.root / "data/processed/logmel/logmel_10s_float16.npy"
        self.done_path = self.root / "data/processed/logmel/logmel_10s_done.npy"
        self.result_dir = self.root / "results/cnn_unseen_revised" / run_id
        self.checkpoint_dir = self.root / "checkpoints/cnn_unseen_revised" / run_id
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
        )
        self.segments = pd.read_csv(self.manifest_path)
        self.index = pd.read_csv(self.index_path)
        self.manifest_hash = sha256_file(self.manifest_path)
        self.index_hash = sha256_file(self.index_path)
        self.protocol_hash = sha256_file(self.root / "docs/ROBUSTNESS_PROTOCOL.md")
        if self.protocol_hash != PROTOCOL_SHA256:
            raise ValueError("Predeclared robustness protocol changed before this run")
        self._validate_inputs()
        self.config_path = self.result_dir / "run_config.json"
        self._save_run_config()

    def _validate_inputs(self) -> None:
        """Stop on split/group/class/cache inconsistencies before fitting."""
        frame = self.segments
        required = {
            "segment_id",
            "track_sample_id",
            "original_audio",
            "label",
            "label_id",
            "split",
            "generator",
        }
        if not required.issubset(frame.columns):
            raise ValueError(
                f"Missing manifest fields: {sorted(required-set(frame.columns))}"
            )
        if (
            frame.segment_id.duplicated().any()
            or self.index.segment_id.duplicated().any()
        ):
            raise ValueError("Duplicate segment_id in manifest or cache index")
        if set(frame.split) != {"train", "val", "test"}:
            raise ValueError("Unexpected split values")
        if ((frame.label == "REAL") != (frame.label_id == 0)).any() or set(
            frame.label_id
        ) != {0, 1}:
            raise ValueError("REAL=0 / FAKE=1 mapping mismatch")
        assert_group_disjoint(
            frame.rename(columns={"original_audio": "original_audio_id"})
        )
        if (frame.groupby("track_sample_id").label_id.nunique() > 1).any():
            raise ValueError("Track contains mixed labels")
        if not frame.segment_id.reset_index(drop=True).equals(
            self.index.segment_id.reset_index(drop=True)
        ):
            raise ValueError("Manifest/cache segment order differs")
        if not np.array_equal(
            self.index.logmel_index.to_numpy(), np.arange(len(frame))
        ):
            raise ValueError("Log-Mel cache index differs from manifest row order")
        if not np.load(self.done_path).all():
            raise ValueError("Log-Mel cache is incomplete")
        cache = np.load(self.cache_path, mmap_mode="r")
        if cache.shape != (len(frame), 128, 1001):
            raise ValueError(f"Unexpected existing Log-Mel shape: {cache.shape}")

    def _save_run_config(self) -> None:
        """Freeze search and Test cohort rules before opening target Test scores."""
        search = {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in SEARCH.items()
        }
        info = dict(
            run_id=self.run_id,
            seed=SEED,
            device=str(self.device),
            holdouts=list(HOLDOUTS),
            baseline=BASELINE,
            search=search,
            mel_settings=MEL_SETTINGS,
            manifest_sha256=self.manifest_hash,
            cache_index_sha256=self.index_hash,
            protocol_sha256=self.protocol_hash,
            versions={
                "python": platform.python_version(),
                "torch": version("torch"),
                "numpy": version("numpy"),
                "pandas": version("pandas"),
                "scikit_learn": version("scikit-learn"),
            },
            test_cohort_rule="all original Test REAL tracks plus target-generator FAKE Test tracks",
            prior_exposure="Historical notebook 14 previously reported MusicGen/Udio Test results; this is not a never-exposed blind Test.",
            selection_rule="raw Validation Track EER asc, Track AUC desc, candidate order asc",
        )
        if self.config_path.exists():
            old = json.loads(self.config_path.read_text())
            if old != info:
                raise ValueError("Existing run ID has different frozen config")
        else:
            self.config_path.write_text(
                json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    # Train·Validation에서 목표 생성기 FAKE만 제외하고 원래 split은 유지한다.
    def filtered_data(self, holdout: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Keep split membership fixed while removing target FAKE from fitting/selection."""
        if holdout not in HOLDOUTS:
            raise ValueError(f"Unknown holdout: {holdout}")
        frame = self.segments
        included = (frame.label_id == 0) | (frame.generator != holdout)
        train = frame.loc[(frame.split == "train") & included].copy()
        val = frame.loc[(frame.split == "val") & included].copy()
        for name, part in (("Train", train), ("Validation", val)):
            if (part.generator == holdout).any() or set(part.label_id) != {0, 1}:
                raise ValueError(
                    f"{name} target leakage or missing class for {holdout}"
                )
        if set(train.original_audio) & set(val.original_audio):
            raise ValueError("Train/Validation original_audio overlap")
        return train, val

    def _loader(self, frame: pd.DataFrame, batch_size: int, shuffle: bool):
        """Build a fixed-ID DataLoader; only Train is shuffled."""
        dataset = LogMelDataset(frame, self.cache_path, self.index)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
            generator=torch.Generator().manual_seed(SEED) if shuffle else None,
        )

    def _train_epoch(self, model, loader, criterion, optimizer, expected_n: int):
        """Train one epoch and return sample-weighted loss plus optimizer updates."""
        model.train()
        weighted_loss = 0.0
        seen = updates = 0
        for xb, yb, _ in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            # 1. 이전 batch의 gradient를 초기화한다.
            optimizer.zero_grad(set_to_none=True)
            # 2. Forward: Log-Mel tensor에서 FAKE logit을 계산한다.
            logits = model(xb)
            # 3. Loss: Train-only pos_weight를 적용한 BCE를 계산한다.
            loss = criterion(logits, yb)
            # 4. Backward: 각 parameter의 gradient를 계산한다.
            loss.backward()
            # 5. Optimizer step: AdamW로 CNN 가중치를 갱신한다.
            optimizer.step()
            batch_n = len(xb)
            weighted_loss += float(loss.item()) * batch_n
            seen += batch_n
            updates += 1
        if seen != expected_n:
            raise ValueError("A Train segment was silently dropped")
        return weighted_loss / seen, updates

    def _score(self, model, loader, frame: pd.DataFrame):
        """Evaluate all fixed segment IDs with the shared binary evaluator."""
        model.eval()
        start = time.perf_counter()
        # Validation/Test 추론에서는 gradient가 필요하지 않다.
        with torch.no_grad():
            scores = predict_scores(model, loader, frame, self.device)
        inference_sec = time.perf_counter() - start
        if set(scores.segment_id) != set(frame.segment_id):
            raise ValueError("Scored segment set differs from fixed inclusion mask")
        return scores, evaluate_binary_predictions(scores), inference_sec

    def _trial_paths(self, holdout: str, trial_id: str):
        folder = self.result_dir / holdout
        folder.mkdir(parents=True, exist_ok=True)
        return (
            self.checkpoint_dir / f"{trial_id}.pt",
            folder / f"{trial_id}_val_scores.csv",
        )

    # 한 후보를 새로 학습하며 최고 epoch 선택은 필터된 Validation만 사용한다.
    # 입력: 한 생성기를 제외한 Train/Validation, 후보 설정과 순서.
    # 출력: 후보의 Validation 지표 행과 epoch별 기록; 최고 epoch checkpoint는 별도 저장.
    def _run_trial(
        self,
        holdout: str,
        trial_id: str,
        stage: str,
        config: dict,
        order: int,
        train: pd.DataFrame,
        val: pd.DataFrame,
    ):
        """Train one independent trial and restore its best Validation epoch."""
        seed_everything()
        train_loader = self._loader(train, config["batch_size"], True)
        val_loader = self._loader(val, config["batch_size"], False)
        model = LogMelCNN(config["dropout"]).to(self.device)
        train_real, train_fake = int((train.label_id == 0).sum()), int(
            (train.label_id == 1).sum()
        )
        pos_weight = train_real / train_fake
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=self.device)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
        )
        checkpoint_path, score_path = self._trial_paths(holdout, trial_id)
        best_key = None
        best_epoch = None
        stale = updates_total = 0
        history = []
        start = time.perf_counter()
        for epoch in range(1, config["max_epochs"] + 1):
            epoch_start = time.perf_counter()
            train_loss, updates = self._train_epoch(
                model, train_loader, criterion, optimizer, len(train)
            )
            updates_total += updates
            val_scores, report, inference_sec = self._score(model, val_loader, val)
            key = (report["track"]["eer"], -report["track"]["roc_auc"], epoch)
            if not np.isfinite(key[0]) or not np.isfinite(key[1]):
                raise ValueError("Validation Track EER/AUC is undefined")
            improved = best_key is None or key < best_key
            if improved:
                best_key, best_epoch, stale = key, epoch, 0
                torch.save(
                    dict(
                        model_state_dict=model.state_dict(),
                        config=config,
                        epoch=epoch,
                        holdout_generator=holdout,
                        run_id=self.run_id,
                        seed=SEED,
                        train_segment_sha256=id_hash(train),
                        val_segment_sha256=id_hash(val),
                        manifest_sha256=self.manifest_hash,
                        cache_index_sha256=self.index_hash,
                        protocol_sha256=self.protocol_hash,
                        mel_settings=MEL_SETTINGS,
                        pos_weight=pos_weight,
                        val_track_eer=report["track"]["eer"],
                        val_track_roc_auc=report["track"]["roc_auc"],
                    ),
                    checkpoint_path,
                )
            else:
                stale += 1
            history.append(
                dict(
                    run_id=self.run_id,
                    holdout_generator=holdout,
                    trial_id=trial_id,
                    stage=stage,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_segment_roc_auc=report["segment"]["roc_auc"],
                    val_segment_eer=report["segment"]["eer"],
                    val_track_roc_auc=report["track"]["roc_auc"],
                    val_track_eer=report["track"]["eer"],
                    is_best_epoch=improved,
                    early_stop_counter=stale,
                    updates=updates,
                    cumulative_updates=updates_total,
                    val_inference_sec=inference_sec,
                    epoch_sec=time.perf_counter() - epoch_start,
                )
            )
            print(
                f"{holdout}/{trial_id} epoch {epoch:02d}: loss={train_loss:.4f}, "
                f"Val Track EER={key[0]:.4f}, AUC={-key[1]:.4f}, best={improved}, stale={stale}",
                flush=True,
            )
            if stale >= config["patience"]:
                break
        saved = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        model.load_state_dict(saved["model_state_dict"])
        val_scores, report, inference_sec = self._score(model, val_loader, val)
        val_scores.assign(
            run_id=self.run_id,
            holdout_generator=holdout,
            trial_id=trial_id,
            model="Log-Mel CNN",
            split="val",
            segment_threshold=report["thresholds"]["segment"],
            track_threshold=report["thresholds"]["track"],
        ).to_csv(score_path, index=False)
        row = dict(
            run_id=self.run_id,
            holdout_generator=holdout,
            model="Log-Mel CNN",
            split="val",
            level="segment+track",
            trial_id=trial_id,
            stage=stage,
            candidate_order=order,
            status="complete",
            **config,
            best_epoch=best_epoch,
            epochs_run=len(history),
            updates=updates_total,
            elapsed_sec=time.perf_counter() - start,
            val_inference_sec=inference_sec,
            device=str(self.device),
            checkpoint=str(checkpoint_path),
            val_scores_path=str(score_path),
            train_real=train_real,
            train_fake=train_fake,
            pos_weight=pos_weight,
            n_train_segments=len(train),
            n_val_segments=len(val),
            train_segment_sha256=id_hash(train),
            val_segment_sha256=id_hash(val),
            segment_roc_auc=report["segment"]["roc_auc"],
            segment_eer=report["segment"]["eer"],
            track_roc_auc=report["track"]["roc_auc"],
            track_eer=report["track"]["eer"],
            track_macro_f1=report["track"]["macro_f1"],
            segment_threshold=report["thresholds"]["segment"],
            track_threshold=report["thresholds"]["track"],
        )
        return row, history

    # 과거 36회 탐색 진입점은 사용자 변경 지시에 따라 실행을 막는다.
    def tune_holdout(self, holdout: str) -> dict:
        """Archived search entry point; superseded by fixed-transfer protocol."""
        raise RuntimeError(
            "The 36-trial unseen CNN search was aborted by user instruction; use FixedTransferCNNExperiment"
        )
        train, val = self.filtered_data(holdout)
        folder = self.result_dir / holdout
        folder.mkdir(parents=True, exist_ok=True)
        tuning_path = folder / "cnn_tuning.csv"
        history_path = folder / "cnn_epoch_history.csv"
        rows = (
            pd.read_csv(tuning_path).to_dict("records") if tuning_path.exists() else []
        )
        history_rows = (
            pd.read_csv(history_path).to_dict("records")
            if history_path.exists()
            else []
        )
        if any(
            row["run_id"] != self.run_id or row["holdout_generator"] != holdout
            for row in rows
        ):
            raise ValueError("Cannot resume a different run/holdout")

        def candidate(trial_id, stage, config, order):
            existing = [row for row in rows if row["trial_id"] == trial_id]
            if existing:
                row = existing[0]
                if (
                    row["status"] != "complete"
                    or not Path(row["checkpoint"]).exists()
                    or not Path(row["val_scores_path"]).exists()
                ):
                    raise ValueError(f"Incomplete saved trial: {trial_id}")
                if int(row["candidate_order"]) != order or row["stage"] != stage:
                    raise ValueError(f"Saved candidate order/stage changed: {trial_id}")
                for key, value in config.items():
                    if row[key] != value:
                        raise ValueError(
                            f"Saved candidate config changed: {trial_id}/{key}"
                        )
                return row
            try:
                row, history = self._run_trial(
                    holdout, trial_id, stage, config, order, train, val
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    failed = dict(
                        run_id=self.run_id,
                        holdout_generator=holdout,
                        trial_id=trial_id,
                        stage=stage,
                        candidate_order=order,
                        status="oom",
                        error=str(exc)[:500],
                    )
                    pd.DataFrame(rows + [failed]).to_csv(tuning_path, index=False)
                raise
            rows.append(row)
            history_rows.extend(history)
            pd.DataFrame(rows).sort_values("candidate_order").to_csv(
                tuning_path, index=False
            )
            pd.DataFrame(history_rows).to_csv(history_path, index=False)
            if self.device.type == "mps":
                torch.mps.empty_cache()
            return row

        baseline = candidate(f"{holdout}_baseline", "baseline", BASELINE, 0)
        stage1_order = 0
        for lr in SEARCH["lr"]:
            for dropout in SEARCH["dropout"]:
                stage1_order += 1
                config = {
                    **BASELINE,
                    "lr": lr,
                    "dropout": dropout,
                    "max_epochs": SEARCH["max_epochs"],
                }
                candidate(
                    f"{holdout}_s1_{stage1_order:02d}", "stage1", config, stage1_order
                )
        stage1_best = select_best_candidate(
            [row for row in rows if row["stage"] == "stage1"]
        )
        seen = {
            (r["lr"], r["dropout"], r["batch_size"], r["weight_decay"], r["max_epochs"])
            for r in rows
            if r["stage"] == "stage1"
        }
        stage2_order = 9
        for batch_size in SEARCH["batch_size"]:
            for weight_decay in SEARCH["weight_decay"]:
                config = {
                    **BASELINE,
                    "lr": stage1_best["lr"],
                    "dropout": stage1_best["dropout"],
                    "batch_size": batch_size,
                    "weight_decay": weight_decay,
                    "max_epochs": SEARCH["max_epochs"],
                }
                key = (
                    config["lr"],
                    config["dropout"],
                    batch_size,
                    weight_decay,
                    config["max_epochs"],
                )
                if key in seen:
                    continue
                stage2_order += 1
                candidate(
                    f"{holdout}_s2_{stage2_order:02d}", "stage2", config, stage2_order
                )
                seen.add(key)
        if len(rows) != 18 or any(r["status"] != "complete" for r in rows):
            raise RuntimeError("Strict holdout search did not complete 18 trials")
        best = select_best_candidate(rows)
        chosen = {}
        for role, row in (("baseline", baseline), ("optimized", best)):
            path = self.checkpoint_dir / f"{holdout}_{role}.pt"
            shutil.copy2(row["checkpoint"], path)
            score_path = folder / f"{holdout}_{role}_validation_scores.csv"
            shutil.copy2(row["val_scores_path"], score_path)
            chosen[role] = dict(
                trial=row,
                checkpoint=str(path),
                checkpoint_sha256=sha256_file(path),
                validation_scores=str(score_path),
                thresholds=dict(
                    segment=float(row["segment_threshold"]),
                    track=float(row["track_threshold"]),
                ),
            )
        selection = dict(
            run_id=self.run_id,
            holdout_generator=holdout,
            train_segment_sha256=id_hash(train),
            val_segment_sha256=id_hash(val),
            stage1_best_trial=stage1_best["trial_id"],
            baseline=chosen["baseline"],
            optimized=chosen["optimized"],
            n_trials=len(rows),
            total_trial_sec=float(sum(r["elapsed_sec"] for r in rows)),
        )
        (folder / "selection.json").write_text(
            json.dumps(selection, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        (folder / "validation_thresholds.json").write_text(
            json.dumps(
                {role: item["thresholds"] for role, item in chosen.items()}, indent=2
            ),
            encoding="utf-8",
        )
        (folder / "best_configs.json").write_text(
            json.dumps(
                {role: item["trial"] for role, item in chosen.items()},
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        (folder / "baseline_configs.json").write_text(
            json.dumps(BASELINE, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"{holdout} selection frozen: {best['trial_id']} "
            f"Val Track EER={best['track_eer']:.6f}, AUC={best['track_roc_auc']:.6f}",
            flush=True,
        )
        return selection

    # 두 holdout의 선택 산출물이 모두 갖춰져야 Test 추론 단계로 간다.
    def freeze_both(self) -> dict:
        """Require both independently filtered searches before any Test scoring."""
        frozen = {}
        for holdout in HOLDOUTS:
            path = self.result_dir / holdout / "selection.json"
            if not path.exists():
                raise RuntimeError(f"Missing selection for {holdout}")
            selection = json.loads(path.read_text())
            for role in ("baseline", "optimized"):
                item = selection[role]
                if sha256_file(Path(item["checkpoint"])) != item["checkpoint_sha256"]:
                    raise ValueError("Selected checkpoint changed after Validation")
            frozen[holdout] = selection
        path = self.result_dir / "selections_frozen.json"
        path.write_text(
            json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return frozen

    def _test_frame(self, holdout: str) -> pd.DataFrame:
        """Use all original Test REAL plus target-generator Test FAKE only."""
        frame = self.segments
        subset = frame.loc[
            (frame.split == "test")
            & ((frame.label_id == 0) | (frame.generator == holdout))
        ].copy()
        if not set(subset.label_id) == {0, 1}:
            raise ValueError("Test cohort lacks one class")
        if (subset.loc[subset.label_id == 1, "generator"] != holdout).any():
            raise ValueError("Non-target FAKE in strict unseen Test")
        if subset.loc[subset.label_id == 0, "track_sample_id"].nunique() != 45:
            raise ValueError("Primary REAL Test pool must contain all 45 tracks")
        return subset

    # Test는 동결된 모델의 최종 평가에만 사용하며 여기서 재학습하지 않는다.
    def final_test(self) -> pd.DataFrame:
        """Archived search Test entry point; no old-search Test is authorized."""
        raise RuntimeError("The 36-trial unseen CNN search was aborted before Test")
        frozen_path = self.result_dir / "selections_frozen.json"
        if not frozen_path.exists():
            raise RuntimeError("Both holdouts must freeze before Test scoring")
        frozen = json.loads(frozen_path.read_text())
        if set(frozen) != set(HOLDOUTS):
            raise ValueError("Freeze file lacks one holdout")
        records = []
        raw_frames = []
        results_path = self.result_dir / "primary_test_metrics.csv"
        if results_path.exists():
            self.paired_source_test()
            return pd.read_csv(results_path)
        for holdout in HOLDOUTS:
            test = self._test_frame(holdout)
            train, val = self.filtered_data(holdout)
            fake_groups = set(test.loc[test.label_id == 1, "original_audio"])
            n_unpaired_real_groups = test.loc[
                (test.label_id == 0) & (~test.original_audio.isin(fake_groups)),
                "original_audio",
            ].nunique()
            print(
                f"{holdout} primary Test: {len(test)} segments, "
                f"REAL/FAKE tracks={test.groupby('label_id').track_sample_id.nunique().to_dict()}, "
                f"REAL groups without target FAKE={n_unpaired_real_groups}",
                flush=True,
            )
            for role in ("baseline", "optimized"):
                item = frozen[holdout][role]
                checkpoint_path = Path(item["checkpoint"])
                if sha256_file(checkpoint_path) != item["checkpoint_sha256"]:
                    raise ValueError("Frozen checkpoint changed before Test")
                checkpoint = torch.load(
                    checkpoint_path, map_location=self.device, weights_only=False
                )
                if (
                    checkpoint["holdout_generator"] != holdout
                    or checkpoint["manifest_sha256"] != self.manifest_hash
                ):
                    raise ValueError("Checkpoint holdout/manifest mismatch")
                if checkpoint["protocol_sha256"] != self.protocol_hash:
                    raise ValueError("Checkpoint protocol hash mismatch")
                if checkpoint["train_segment_sha256"] != id_hash(train) or checkpoint[
                    "val_segment_sha256"
                ] != id_hash(val):
                    raise ValueError("Checkpoint inclusion mask mismatch")
                model = LogMelCNN(checkpoint["config"]["dropout"]).to(self.device)
                model.load_state_dict(checkpoint["model_state_dict"])
                loader = self._loader(test, checkpoint["config"]["batch_size"], False)
                start = time.perf_counter()
                with torch.no_grad():
                    frame = predict_scores(model, loader, test, self.device)
                inference_sec = time.perf_counter() - start
                report = evaluate_binary_predictions(frame, item["thresholds"])
                folder = self.result_dir / holdout
                for level in ("segment", "track"):
                    scores = report[f"{level}_scores"].copy()
                    scores = scores.assign(
                        run_id=self.run_id,
                        holdout_generator=holdout,
                        model="Log-Mel CNN",
                        role=role,
                        variant=role,
                        cohort="primary",
                        split="test",
                        level=level,
                        threshold=item["thresholds"][level],
                    )
                    scores.to_csv(
                        folder / f"{holdout}_{role}_test_{level}_scores.csv",
                        index=False,
                    )
                    raw_frames.append(scores)
                    metric = report[level]
                    records.append(
                        dict(
                            run_id=self.run_id,
                            holdout_generator=holdout,
                            model="Log-Mel CNN",
                            role=role,
                            variant=role,
                            cohort="primary",
                            split="test",
                            level=level,
                            validation_segment_threshold=item["thresholds"]["segment"],
                            validation_track_threshold=item["thresholds"]["track"],
                            n=metric["n"],
                            n_real=metric["n_real"],
                            n_fake=metric["n_fake"],
                            real_prevalence=metric["real_prevalence"],
                            fake_prevalence=metric["fake_prevalence"],
                            threshold=metric["threshold"],
                            roc_auc=metric["roc_auc"],
                            ap_fake=metric["ap_fake"],
                            ap_real=metric["ap_real"],
                            eer=metric["eer"],
                            balanced_accuracy=metric["balanced_accuracy"],
                            macro_f1=metric["macro_f1"],
                            real_fpr=metric["real_fpr"],
                            fake_miss_rate=metric["fake_miss_rate"],
                            hter=metric["hter"],
                            reason=metric["reason"],
                            confusion_matrix=json.dumps(metric["confusion_matrix"]),
                            inference_sec=inference_sec,
                            n_unpaired_real_original_groups=n_unpaired_real_groups,
                            checkpoint_sha256=item["checkpoint_sha256"],
                        )
                    )
                print(
                    f"{holdout}/{role} Test Track EER={report['track']['eer']:.6f}, "
                    f"AUC={report['track']['roc_auc']:.6f}, HTER={report['track']['hter']:.6f}",
                    flush=True,
                )
                del model
                if self.device.type == "mps":
                    torch.mps.empty_cache()
        results = pd.DataFrame(records)
        pd.concat(raw_frames, ignore_index=True).to_csv(
            self.result_dir / "primary_test_scores.csv", index=False
        )
        results.to_csv(results_path, index=False)
        self.paired_source_test()
        return results

    def paired_source_test(self) -> pd.DataFrame:
        """Re-evaluate Udio's paired original groups from saved primary scores only."""
        output_path = self.result_dir / "paired_source_test_metrics.csv"
        if output_path.exists():
            return pd.read_csv(output_path)
        frozen_path = self.result_dir / "selections_frozen.json"
        if not frozen_path.exists():
            raise RuntimeError("Selections have not been frozen")
        frozen = json.loads(frozen_path.read_text())
        holdout = "udio"
        folder = self.result_dir / holdout
        records = []
        for role in ("baseline", "optimized"):
            primary_path = folder / f"{holdout}_{role}_test_segment_scores.csv"
            if not primary_path.exists():
                raise RuntimeError(
                    "Primary Test scores must exist before paired-source analysis"
                )
            primary = pd.read_csv(primary_path)
            fake_groups = set(primary.loc[primary.label == 1, "original_audio_id"])
            paired = primary.loc[primary.original_audio_id.isin(fake_groups)].copy()
            report = evaluate_binary_predictions(
                paired, frozen[holdout][role]["thresholds"]
            )
            for level in ("segment", "track"):
                metric = report[level]
                report[f"{level}_scores"].assign(
                    run_id=self.run_id,
                    holdout_generator=holdout,
                    model="Log-Mel CNN",
                    role=role,
                    variant=role,
                    cohort="paired_source",
                    split="test",
                    level=level,
                    threshold=frozen[holdout][role]["thresholds"][level],
                ).to_csv(
                    folder / f"{holdout}_{role}_paired_source_{level}_scores.csv",
                    index=False,
                )
                records.append(
                    dict(
                        run_id=self.run_id,
                        holdout_generator=holdout,
                        model="Log-Mel CNN",
                        role=role,
                        variant=role,
                        cohort="paired_source",
                        split="test",
                        level=level,
                        validation_segment_threshold=frozen[holdout][role][
                            "thresholds"
                        ]["segment"],
                        validation_track_threshold=frozen[holdout][role]["thresholds"][
                            "track"
                        ],
                        **{
                            key: value
                            for key, value in metric.items()
                            if key != "confusion_matrix"
                        },
                        confusion_matrix=json.dumps(metric["confusion_matrix"]),
                        n_source_original_groups=len(fake_groups),
                        source_score_path=str(primary_path),
                    )
                )
        result = pd.DataFrame(records)
        result.to_csv(output_path, index=False)
        return result
