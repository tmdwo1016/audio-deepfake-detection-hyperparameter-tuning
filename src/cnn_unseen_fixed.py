"""Fresh CNN training with main-model hyperparameters fixed across holdouts.

The main experiment selected the architecture/hyperparameters on Validation
that included MusicGen/Udio. That exposure is recorded; no holdout-specific
hyperparameter search is performed here. Each holdout has a fresh model fit
and its own filtered-Validation epoch and thresholds.
"""

from __future__ import annotations

import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import torch

from src.cnn_modeling import LogMelCNN, predict_scores
from src.cnn_unseen_revised import (
    HOLDOUTS,
    MEL_SETTINGS,
    SEED,
    UnseenCNNExperiment,
    id_hash,
    sha256_file,
)
from src.modeling_evaluation import evaluate_binary_predictions

PROTOCOL_SHA256 = "098fa65192f65fddbf7d65757eb5c6b5c2b1149f79178c27746b4e328ee54a5e"
VARIANT = "fixed_hyperparameter_transfer"


class FixedTransferCNNExperiment(UnseenCNNExperiment):
    """Reuse verified CNN fitting primitives with an isolated fixed protocol."""

    def __init__(self, root: Path, run_id: str):
        self.root = Path(root)
        self.run_id = run_id
        self.manifest_path = self.root / "data/metadata/segment_manifest_10s.csv"
        self.index_path = self.root / "data/processed/logmel/logmel_10s_index.csv"
        self.cache_path = self.root / "data/processed/logmel/logmel_10s_float16.npy"
        self.done_path = self.root / "data/processed/logmel/logmel_10s_done.npy"
        self.result_dir = self.root / "results/cnn_unseen_fixed_transfer" / run_id
        self.checkpoint_dir = (
            self.root / "checkpoints/cnn_unseen_fixed_transfer" / run_id
        )
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
        # 사전에 고정한 실험 규칙과 입력 manifest를 해시로 묶어 변경을 감지한다.
        self.protocol_hash = sha256_file(self.root / "docs/UNSEEN_FIXED_PROTOCOL.md")
        if self.protocol_hash != PROTOCOL_SHA256:
            raise ValueError("Fixed-transfer protocol hash changed")
        self._validate_inputs()

        self.main_checkpoint_path = self.root / "checkpoints/optimized/cnn_best.pt"
        self.main_config_path = self.root / "results/model_tuning/cnn_best_config.json"
        self.main_checkpoint_hash = sha256_file(self.main_checkpoint_path)
        self.main_config_hash = sha256_file(self.main_config_path)
        # Main Validation에서 이미 선택한 구조와 설정을 그대로 가져온다.
        # 대상별 Validation으로 학습률·dropout 등을 다시 탐색하지 않는다.
        main = torch.load(
            self.main_checkpoint_path, map_location="cpu", weights_only=False
        )
        config_json = json.loads(self.main_config_path.read_text())
        if (
            main["manifest_sha256"] != self.manifest_hash
            or main["cache_index_sha256"] != self.index_hash
        ):
            raise ValueError("Main selected checkpoint used different manifest/cache")
        if main["mel_settings"] != MEL_SETTINGS:
            raise ValueError("Main selected checkpoint used different Log-Mel settings")
        keys = (
            "lr",
            "dropout",
            "batch_size",
            "weight_decay",
            "max_epochs",
            "patience",
            "optimizer",
            "architecture",
        )
        self.fixed_config = {key: main["config"][key] for key in keys}
        if any(self.fixed_config[key] != config_json["best"][key] for key in keys):
            raise ValueError("Main checkpoint and best-config JSON disagree")
        if self.fixed_config != dict(
            lr=3e-4,
            dropout=0.3,
            batch_size=16,
            weight_decay=1e-3,
            max_epochs=30,
            patience=4,
            optimizer="AdamW",
            architecture="historical_4_conv_blocks",
        ):
            raise ValueError(
                "Main selected hyperparameters differ from predeclared protocol"
            )
        self.config_path = self.result_dir / "run_config.json"
        self._save_fixed_config()

    def _save_fixed_config(self) -> None:
        """Write exact parent model provenance before fitting any target."""
        info = dict(
            run_id=self.run_id,
            protocol=VARIANT,
            protocol_sha256=self.protocol_hash,
            seed=SEED,
            device=str(self.device),
            holdouts=list(HOLDOUTS),
            fixed_config=self.fixed_config,
            mel_settings=MEL_SETTINGS,
            main_checkpoint=str(self.main_checkpoint_path),
            main_checkpoint_sha256=self.main_checkpoint_hash,
            main_best_config_json=str(self.main_config_path),
            main_best_config_sha256=self.main_config_hash,
            manifest_sha256=self.manifest_hash,
            cache_index_sha256=self.index_hash,
            versions={
                "python": platform.python_version(),
                "torch": version("torch"),
                "numpy": version("numpy"),
                "pandas": version("pandas"),
                "scikit_learn": version("scikit-learn"),
            },
            selection_rule="filtered Validation Track EER asc, AUC desc, epoch asc",
            test_cohort_rule="all original Test REAL plus target-generator FAKE",
            prior_exposure="Main hyperparameters used target-generator Validation; historical target Test results were already public.",
        )
        if self.config_path.exists():
            if json.loads(self.config_path.read_text()) != info:
                raise ValueError("Existing fixed-transfer run ID has different config")
        else:
            self.config_path.write_text(
                json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    # 보류 생성기의 FAKE는 Train·Validation에서 빼고 새 모델을 학습한다.
    # 입력: 보류할 생성기 이름. 출력: 새 학습 checkpoint와 Validation 임계값 선택 기록.
    def fit_holdout(self, holdout: str) -> dict:
        """Fit exactly one fresh model with parent hyperparameters for a holdout."""
        # 대상 생성기의 FAKE를 원래 Train과 Validation에서 모두 제거한다.
        train, val = self.filtered_data(holdout)
        folder = self.result_dir / holdout
        folder.mkdir(parents=True, exist_ok=True)
        selection_path = folder / "selection.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            if (
                selection["run_id"] != self.run_id
                or selection["holdout_generator"] != holdout
            ):
                raise ValueError("Saved selection belongs to another run")
            if (
                sha256_file(Path(selection["checkpoint"]))
                != selection["checkpoint_sha256"]
            ):
                raise ValueError("Saved selected checkpoint hash changed")
            if selection["train_segment_sha256"] != id_hash(train) or selection[
                "val_segment_sha256"
            ] != id_hash(val):
                raise ValueError("Saved inclusion mask changed")
            return selection

        trial_id = f"{holdout}_fixed_transfer"
        # 새 CNN 가중치를 학습하며 pos_weight는 이 대상의 Train에서만 계산한다.
        # best epoch와 Segment/Track 임계값은 필터된 Validation으로 선택한다.
        row, history = self._run_trial(
            holdout, trial_id, VARIANT, self.fixed_config, 0, train, val
        )
        pd.DataFrame([row]).to_csv(folder / "training.csv", index=False)
        pd.DataFrame(history).to_csv(folder / "epoch_history.csv", index=False)
        checkpoint = Path(row["checkpoint"])
        selection = dict(
            run_id=self.run_id,
            holdout_generator=holdout,
            variant=VARIANT,
            trial=row,
            checkpoint=str(checkpoint),
            checkpoint_sha256=sha256_file(checkpoint),
            validation_scores=row["val_scores_path"],
            thresholds=dict(
                segment=float(row["segment_threshold"]),
                track=float(row["track_threshold"]),
            ),
            train_segment_sha256=id_hash(train),
            val_segment_sha256=id_hash(val),
            main_checkpoint_sha256=self.main_checkpoint_hash,
            protocol_sha256=self.protocol_hash,
        )
        selection_path.write_text(
            json.dumps(selection, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        (folder / "validation_thresholds.json").write_text(
            json.dumps(selection["thresholds"], indent=2), encoding="utf-8"
        )
        (folder / "fixed_config.json").write_text(
            json.dumps(self.fixed_config, indent=2), encoding="utf-8"
        )
        print(
            f"{holdout} fixed transfer selected epoch {row['best_epoch']}, "
            f"Val Track EER={row['track_eer']:.6f}, AUC={row['track_roc_auc']:.6f}",
            flush=True,
        )
        if self.device.type == "mps":
            torch.mps.empty_cache()
        return selection

    # 두 생성기의 선택·checkpoint·임계값이 모두 저장되기 전에는 Test를 열지 않는다.
    def freeze_both(self) -> dict:
        """Require both fresh fits and threshold files before Test inference."""
        frozen = {}
        for holdout in HOLDOUTS:
            train, val = self.filtered_data(holdout)
            path = self.result_dir / holdout / "selection.json"
            if not path.exists():
                raise RuntimeError(f"Missing fixed-transfer selection for {holdout}")
            selection = json.loads(path.read_text())
            if (
                selection["protocol_sha256"] != self.protocol_hash
                or selection["main_checkpoint_sha256"] != self.main_checkpoint_hash
            ):
                raise ValueError(
                    "Fixed protocol/main hyperparameter provenance changed"
                )
            if selection["train_segment_sha256"] != id_hash(train) or selection[
                "val_segment_sha256"
            ] != id_hash(val):
                raise ValueError("Selected Train/Validation mask changed")
            if (
                sha256_file(Path(selection["checkpoint"]))
                != selection["checkpoint_sha256"]
            ):
                raise ValueError("Selected checkpoint changed")
            frozen[holdout] = selection
        path = self.result_dir / "selections_frozen.json"
        path.write_text(
            json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return frozen

    def final_test(self) -> pd.DataFrame:
        """Score each original Test cohort once using frozen target-specific models."""
        results_path = self.result_dir / "primary_test_metrics.csv"
        if results_path.exists():
            self.paired_source_test()
            return pd.read_csv(results_path)
        frozen_path = self.result_dir / "selections_frozen.json"
        if not frozen_path.exists():
            raise RuntimeError("Both fixed-transfer selections must freeze before Test")
        frozen = json.loads(frozen_path.read_text())
        if set(frozen) != set(HOLDOUTS):
            raise ValueError("Frozen file lacks a target")
        records, raw_frames = [], []
        for holdout in HOLDOUTS:
            test = self._test_frame(holdout)
            train, val = self.filtered_data(holdout)
            item = frozen[holdout]
            checkpoint_path = Path(item["checkpoint"])
            if sha256_file(checkpoint_path) != item["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed before Test")
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            if (
                checkpoint["holdout_generator"] != holdout
                or checkpoint["protocol_sha256"] != self.protocol_hash
            ):
                raise ValueError("Checkpoint holdout/protocol mismatch")
            if (
                checkpoint["manifest_sha256"] != self.manifest_hash
                or checkpoint["cache_index_sha256"] != self.index_hash
            ):
                raise ValueError("Checkpoint input hashes changed")
            if checkpoint["train_segment_sha256"] != id_hash(train) or checkpoint[
                "val_segment_sha256"
            ] != id_hash(val):
                raise ValueError("Checkpoint Train/Validation inclusion masks changed")
            if checkpoint["config"] != self.fixed_config:
                raise ValueError("Checkpoint hyperparameters changed")
            model = LogMelCNN(self.fixed_config["dropout"]).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            loader = self._loader(test, self.fixed_config["batch_size"], False)
            start = time.perf_counter()
            with torch.no_grad():
                scores = predict_scores(model, loader, test, self.device)
            inference_sec = time.perf_counter() - start
            # Test 분류에는 해당 생성기의 Validation 임계값을 그대로 적용한다.
            report = evaluate_binary_predictions(scores, item["thresholds"])
            fake_groups = set(test.loc[test.label_id == 1, "original_audio"])
            n_unpaired_real_groups = test.loc[
                (test.label_id == 0) & (~test.original_audio.isin(fake_groups)),
                "original_audio",
            ].nunique()
            folder = self.result_dir / holdout
            for level in ("segment", "track"):
                level_scores = (
                    report[f"{level}_scores"]
                    .copy()
                    .assign(
                        run_id=self.run_id,
                        holdout_generator=holdout,
                        model="Log-Mel CNN",
                        variant=VARIANT,
                        cohort="primary",
                        split="test",
                        level=level,
                        threshold=item["thresholds"][level],
                    )
                )
                level_scores.to_csv(
                    folder / f"{holdout}_test_{level}_scores.csv", index=False
                )
                raw_frames.append(level_scores)
                metric = report[level]
                records.append(
                    dict(
                        run_id=self.run_id,
                        holdout_generator=holdout,
                        model="Log-Mel CNN",
                        variant=VARIANT,
                        cohort="primary",
                        split="test",
                        level=level,
                        validation_segment_threshold=item["thresholds"]["segment"],
                        validation_track_threshold=item["thresholds"]["track"],
                        **{
                            key: value
                            for key, value in metric.items()
                            if key != "confusion_matrix"
                        },
                        confusion_matrix=json.dumps(metric["confusion_matrix"]),
                        inference_sec=inference_sec,
                        n_unpaired_real_original_groups=n_unpaired_real_groups,
                        checkpoint_sha256=item["checkpoint_sha256"],
                    )
                )
            print(
                f"{holdout} fixed-transfer Test Track EER={report['track']['eer']:.6f}, "
                f"AUC={report['track']['roc_auc']:.6f}, HTER={report['track']['hter']:.6f}",
                flush=True,
            )
            del model
            if self.device.type == "mps":
                torch.mps.empty_cache()
        result = pd.DataFrame(records)
        pd.concat(raw_frames, ignore_index=True).to_csv(
            self.result_dir / "primary_test_scores.csv", index=False
        )
        result.to_csv(results_path, index=False)
        self.paired_source_test()
        return result

    def paired_source_test(self) -> pd.DataFrame:
        """Optional Udio paired-source sensitivity from saved primary scores."""
        path = self.result_dir / "paired_source_test_metrics.csv"
        if path.exists():
            return pd.read_csv(path)
        frozen_path = self.result_dir / "selections_frozen.json"
        if not frozen_path.exists():
            raise RuntimeError("Both selections must freeze before paired analysis")
        frozen = json.loads(frozen_path.read_text())
        holdout = "udio"
        folder = self.result_dir / holdout
        primary_path = folder / f"{holdout}_test_segment_scores.csv"
        if not primary_path.exists():
            raise RuntimeError("Primary Udio Test score file missing")
        primary = pd.read_csv(primary_path)
        fake_groups = set(primary.loc[primary.label == 1, "original_audio_id"])
        paired = primary.loc[primary.original_audio_id.isin(fake_groups)].copy()
        thresholds = frozen[holdout]["thresholds"]
        report = evaluate_binary_predictions(paired, thresholds)
        records = []
        for level in ("segment", "track"):
            metric = report[level]
            report[f"{level}_scores"].assign(
                run_id=self.run_id,
                holdout_generator=holdout,
                model="Log-Mel CNN",
                variant=VARIANT,
                cohort="paired_source",
                split="test",
                level=level,
                threshold=thresholds[level],
            ).to_csv(
                folder / f"{holdout}_paired_source_{level}_scores.csv", index=False
            )
            records.append(
                dict(
                    run_id=self.run_id,
                    holdout_generator=holdout,
                    model="Log-Mel CNN",
                    variant=VARIANT,
                    cohort="paired_source",
                    split="test",
                    level=level,
                    validation_segment_threshold=thresholds["segment"],
                    validation_track_threshold=thresholds["track"],
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
        result.to_csv(path, index=False)
        return result
