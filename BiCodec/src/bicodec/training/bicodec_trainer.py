from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from bicodec.data.collate import collate_audio_batch
from bicodec.data.dataset import AudioManifestDataset
from bicodec.training.losses import (
    DiscriminatorLoss,
    FeatureMatchingLoss,
    GeneratorLoss,
    MelSpecReconstructionLoss,
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    MultiResolutionSTFTLoss,
    average_discriminator_loss,
    average_feature_matching_loss,
    average_generator_loss,
    compute_reconstruction_losses,
    crop_to_match_last_dim,
)
from bicodec.training.setup import (
    FrozenWav2Vec2Frontend,
    build_bicodec_model,
    cleanup_ddp,
    ddp_average_metrics,
    get_rank,
    get_world_size,
    initialize_training_model,
    is_dist_initialized,
    is_main_process,
    make_optimizer,
    maybe_load_resume,
    resolve_training_paths,
    save_checkpoint,
    set_bicodec_experiment_model_train_mode,
    set_seed,
    setup_device_and_ddp,
    unwrap_model,
)


def _build_manifest_dataset(
    cfg: DictConfig,
    manifest_path: str,
    sample_rate: int,
    model_config: Any,
    speaker_id_to_idx: Optional[Dict[str, int]],
) -> AudioManifestDataset:
    ref_segment_duration = float(
        model_config.get("ref_segment_duration", cfg.data.get("ref_segment_duration", 3.0))
    )
    return AudioManifestDataset(
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        audio_key=cfg.data.get("audio_key", "audio_filepath"),
        base_dir=cfg.data.get("base_dir"),
        volume_normalize=bool(cfg.data.get("volume_normalize", False)),
        segment_duration=cfg.data.get("segment_duration"),
        min_duration=cfg.data.get("min_duration"),
        max_duration=cfg.data.get("max_duration"),
        ref_segment_duration=ref_segment_duration,
        speaker_id_key=str(cfg.data.get("speaker_id_key", "speaker_id")),
        speaker_id_to_idx=speaker_id_to_idx,
    )


def _build_dataloader(
    dataset: AudioManifestDataset,
    cfg: DictConfig,
    device: torch.device,
    *,
    batch_size: int,
    sampler: Optional[DistributedSampler],
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=int(cfg.trainer.num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=collate_audio_batch,
        drop_last=False,
        persistent_workers=int(cfg.trainer.num_workers) > 0,
    )


def _discriminator_state(
    mpd: nn.Module,
    mrd: nn.Module,
    optimizer_d: torch.optim.Optimizer,
    scheduler_d: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler_d: GradScaler,
) -> Dict[str, Any]:
    return {
        "mpd": unwrap_model(mpd).state_dict(),
        "mrd": unwrap_model(mrd).state_dict(),
        "optimizer_d": optimizer_d.state_dict(),
        "scheduler_d": None if scheduler_d is None else scheduler_d.state_dict(),
        "scaler_d": scaler_d.state_dict(),
    }


def evaluate(
    model: nn.Module,
    frontend: FrozenWav2Vec2Frontend,
    dataloader: DataLoader,
    mel_loss_fn: MelSpecReconstructionLoss,
    stft_loss_fn: MultiResolutionSTFTLoss,
    cfg: DictConfig,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            wav = batch["wav"].to(device, non_blocking=True)
            ref_wav = batch["ref_wav"].to(device, non_blocking=True)
            feat = frontend(wav)

            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model({"wav": wav, "ref_wav": ref_wav, "feat": feat})
                losses = compute_reconstruction_losses(outputs, feat, mel_loss_fn, stft_loss_fn, cfg)

            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            count += 1

    set_bicodec_experiment_model_train_mode(model, cfg)
    local_metrics = {key: value / max(count, 1) for key, value in totals.items()}
    return ddp_average_metrics(local_metrics, device)


def train_bicodec(cfg: DictConfig) -> None:
    set_seed(int(cfg.get("seed", 42)))
    device, local_rank, use_ddp = setup_device_and_ddp(cfg)
    paths = resolve_training_paths(cfg)

    output_dir = Path(cfg.trainer.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "train_config.yaml")

    if is_dist_initialized():
        dist.barrier()

    model, model_config = build_bicodec_model(
        config_path=paths["model_config_path"],
        init_ckpt_path=paths["init_ckpt_path"],
        strict_load=bool(cfg.model.get("strict_load", False)),
        quantizer_distance_loss_type=cfg.loss.get("vq_distance_loss_type"),
        use_codebook=cfg.model.get("use_codebook"),
        speaker_input_key=cfg.model.get("speaker_input_key"),
        speaker_time_condition=cfg.model.get(
            "speaker_time_adapter",
            cfg.model.get("speaker_time_condition"),
        ),
    )
    model = model.to(device)
    use_speaker_cls = bool(cfg.trainer.get("speaker_classification", False))
    speaker_id_to_idx = initialize_training_model(
        cfg,
        model,
        model_config,
        device,
        do_print=is_main_process(),
    )

    frontend = FrozenWav2Vec2Frontend(
        model_name_or_path=paths["feature_extractor_path"],
        hidden_state_indices=cfg.feature_extractor.hidden_state_indices,
        device=device,
    )

    sample_rate = int(
        model_config.get(
            "sample_rate",
            model_config.get("audio_tokenizer", {}).get("sample_rate", 16000),
        )
    )
    if sample_rate != 16000:
        raise ValueError(f"Training frontend currently expects 16000 Hz audio, got {sample_rate}")

    train_dataset = _build_manifest_dataset(
        cfg,
        cfg.data.train_manifest,
        sample_rate,
        model_config,
        speaker_id_to_idx,
    )

    valid_dataset = None
    valid_manifest = cfg.data.get("valid_manifest")
    if valid_manifest:
        valid_dataset = _build_manifest_dataset(
            cfg,
            valid_manifest,
            sample_rate,
            model_config,
            speaker_id_to_idx,
        )

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    valid_sampler = (
        DistributedSampler(valid_dataset, shuffle=False)
        if use_ddp and valid_dataset is not None
        else None
    )

    train_loader = _build_dataloader(
        train_dataset,
        cfg,
        device,
        batch_size=int(cfg.trainer.batch_size),
        sampler=train_sampler,
        shuffle=True,
    )

    valid_loader = None
    if valid_dataset is not None:
        valid_loader = _build_dataloader(
            valid_dataset,
            cfg,
            device,
            batch_size=int(cfg.trainer.eval_batch_size),
            sampler=valid_sampler,
            shuffle=False,
        )

    mel_loss_fn = MelSpecReconstructionLoss(
        sample_rate=sample_rate,
        n_fft=int(cfg.loss.mel_n_fft),
        hop_length=int(cfg.loss.mel_hop_length),
        n_mels=int(cfg.loss.mel_n_mels),
        f_min=int(cfg.loss.get("mel_f_min", 0)),
        f_max=int(cfg.loss.get("mel_f_max", sample_rate // 2)),
    ).to(device)
    stft_loss_fn = MultiResolutionSTFTLoss(cfg.loss.stft_resolutions)

    mpd = MultiPeriodDiscriminator(
        periods=tuple(cfg.discriminator.get("periods", [2, 3, 5, 7, 11]))
    ).to(device)
    mrd = MultiResolutionDiscriminator(
        fft_sizes=tuple(cfg.discriminator.get("fft_sizes", [2048, 1024, 512]))
    ).to(device)

    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        mpd = DDP(mpd, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        mrd = DDP(mrd, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    generator_loss_fn = GeneratorLoss()
    discriminator_loss_fn = DiscriminatorLoss()
    feature_matching_loss_fn = FeatureMatchingLoss()

    optimizer = make_optimizer(cfg, model.parameters())
    disc_optim_cfg = cfg.get("disc_optim", cfg.optim)
    optimizer_d = AdamW(
        list(mpd.parameters()) + list(mrd.parameters()),
        lr=float(disc_optim_cfg.lr),
        betas=tuple(disc_optim_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(disc_optim_cfg.get("weight_decay", 0.0)),
    )

    max_steps = int(cfg.trainer.max_steps)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max_steps,
        eta_min=float(cfg.optim.get("min_lr", 0.0)),
    )
    scheduler_d = CosineAnnealingLR(
        optimizer_d,
        T_max=max_steps,
        eta_min=float(disc_optim_cfg.get("min_lr", 0.0)),
    )

    use_amp = bool(cfg.trainer.get("use_amp", True)) and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)
    scaler_d = GradScaler(device.type, enabled=use_amp)

    start_epoch, global_step, best_val_loss = maybe_load_resume(
        cfg.trainer.get("resume_from"),
        model,
        optimizer,
        scheduler,
        scaler,
        mpd=mpd,
        mrd=mrd,
        optimizer_d=optimizer_d,
        scheduler_d=scheduler_d,
        scaler_d=scaler_d,
    )

    grad_accum_steps = int(cfg.trainer.get("grad_accum_steps", 1))
    log_every = int(cfg.trainer.get("log_every_steps", 10))
    save_every = int(cfg.trainer.get("save_every_steps", 1000))
    eval_every = int(cfg.trainer.get("eval_every_steps", save_every))
    max_epochs = int(cfg.trainer.get("max_epochs", 1))
    pretrain_mel_steps = int(cfg.trainer.get("pretrain_mel_steps", 0))

    if is_main_process():
        print(
            f"[train] rank={get_rank()} world_size={get_world_size()} "
            f"device={device} train_items={len(train_dataset)} "
            f"valid_items={0 if valid_dataset is None else len(valid_dataset)}"
        )
        print(
            f"[train] output_dir={output_dir}  "
            f"speaker_classification={use_speaker_cls}  "
            f"detach_xvector={bool(cfg.trainer.get('detach_xvector_condition', False))}"
        )

    set_bicodec_experiment_model_train_mode(model, cfg)
    optimizer.zero_grad(set_to_none=True)
    optimizer_d.zero_grad(set_to_none=True)

    try:
        for epoch in range(start_epoch, max_epochs):
            if use_ddp and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            for batch in train_loader:
                wav = batch["wav"].to(device, non_blocking=True)
                ref_wav = batch["ref_wav"].to(device, non_blocking=True)
                feat = frontend(wav)
                train_discriminator = global_step >= pretrain_mel_steps

                disc_loss = wav.new_tensor(0.0)
                speaker_cls_loss = wav.new_tensor(0.0)

                if train_discriminator:
                    with torch.no_grad():
                        disc_outputs = model({"wav": wav, "ref_wav": ref_wav, "feat": feat})
                        fake_wav = crop_to_match_last_dim(
                            disc_outputs["recons"],
                            disc_outputs["audios"],
                        )[0].squeeze(1)
                        real_wav = crop_to_match_last_dim(
                            disc_outputs["recons"],
                            disc_outputs["audios"],
                        )[1].squeeze(1)

                    with autocast(device_type=device.type, enabled=use_amp):
                        real_score_mp, fake_score_mp, _, _ = mpd(real_wav, fake_wav)
                        real_score_mrd, fake_score_mrd, _, _ = mrd(real_wav, fake_wav)
                        loss_disc_mp = average_discriminator_loss(
                            discriminator_loss_fn,
                            real_score_mp,
                            fake_score_mp,
                        )
                        loss_disc_mrd = average_discriminator_loss(
                            discriminator_loss_fn,
                            real_score_mrd,
                            fake_score_mrd,
                        )
                        disc_loss = (
                            float(cfg.loss.discriminator_mp_weight) * loss_disc_mp
                            + float(cfg.loss.discriminator_mrd_weight) * loss_disc_mrd
                        )
                        disc_loss_scaled = disc_loss / grad_accum_steps

                    scaler_d.scale(disc_loss_scaled).backward()

                    if (global_step + 1) % grad_accum_steps == 0:
                        scaler_d.unscale_(optimizer_d)
                        torch.nn.utils.clip_grad_norm_(
                            list(mpd.parameters()) + list(mrd.parameters()),
                            float(cfg.trainer.get("clip_grad_norm", 1.0)),
                        )
                        scaler_d.step(optimizer_d)
                        scaler_d.update()
                        optimizer_d.zero_grad(set_to_none=True)
                        scheduler_d.step()

                with autocast(device_type=device.type, enabled=use_amp):
                    outputs = model({"wav": wav, "ref_wav": ref_wav, "feat": feat})
                    recon_losses = compute_reconstruction_losses(
                        outputs,
                        feat,
                        mel_loss_fn,
                        stft_loss_fn,
                        cfg,
                    )
                    pred_wav, target_wav = crop_to_match_last_dim(outputs["recons"], outputs["audios"])
                    fake_wav = pred_wav.squeeze(1)
                    real_wav = target_wav.squeeze(1)

                    loss_gen_mp = fake_wav.new_tensor(0.0)
                    loss_gen_mrd = fake_wav.new_tensor(0.0)
                    loss_fm_mp = fake_wav.new_tensor(0.0)
                    loss_fm_mrd = fake_wav.new_tensor(0.0)

                    if train_discriminator:
                        _, gen_score_mp, fmap_rs_mp, fmap_gs_mp = mpd(real_wav, fake_wav)
                        _, gen_score_mrd, fmap_rs_mrd, fmap_gs_mrd = mrd(real_wav, fake_wav)
                        loss_gen_mp = average_generator_loss(generator_loss_fn, gen_score_mp)
                        loss_gen_mrd = average_generator_loss(generator_loss_fn, gen_score_mrd)
                        loss_fm_mp = average_feature_matching_loss(
                            feature_matching_loss_fn,
                            fmap_rs_mp,
                            fmap_gs_mp,
                        )
                        loss_fm_mrd = average_feature_matching_loss(
                            feature_matching_loss_fn,
                            fmap_rs_mrd,
                            fmap_gs_mrd,
                        )

                    total_gen_loss = (
                        recon_losses["total"]
                        + float(cfg.loss.generator_mp_weight) * loss_gen_mp
                        + float(cfg.loss.generator_mrd_weight) * loss_gen_mrd
                        + float(cfg.loss.feature_matching_mp_weight) * loss_fm_mp
                        + float(cfg.loss.feature_matching_mrd_weight) * loss_fm_mrd
                    )
                    if use_speaker_cls and "speaker_logits" in outputs and "speaker_label" in batch:
                        logits = outputs["speaker_logits"]
                        speaker_labels = batch["speaker_label"].to(device, non_blocking=True)
                        speaker_cls_loss = F.cross_entropy(logits, speaker_labels)
                        total_gen_loss = total_gen_loss + float(
                            cfg.loss.get("speaker_classification_weight", 1.0)
                        ) * speaker_cls_loss
                    loss = total_gen_loss / grad_accum_steps

                scaler.scale(loss).backward()

                if (global_step + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(cfg.trainer.get("clip_grad_norm", 1.0)),
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                global_step += 1

                if global_step % log_every == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    log_payload = {
                        "loss": float(total_gen_loss.detach()),
                        "mel": float(recon_losses["mel"]),
                        "wav_l1": float(recon_losses["waveform_l1"]),
                        "feat_l1": float(recon_losses["feature_l1"]),
                        "stft": float(recon_losses["stft"]),
                        "vq": float(recon_losses["vq"]),
                        "gan_mp": float(loss_gen_mp.detach()),
                        "gan_mrd": float(loss_gen_mrd.detach()),
                        "fm_mp": float(loss_fm_mp.detach()),
                        "fm_mrd": float(loss_fm_mrd.detach()),
                        "disc": float(disc_loss.detach()) if train_discriminator else 0.0,
                        "ppl": float(recon_losses["perplexity"]),
                        "active": float(recon_losses["active_num"]),
                        "spk_ce": float(speaker_cls_loss.detach()),
                    }
                    log_payload = ddp_average_metrics(log_payload, device)

                    if is_main_process():
                        print(
                            "[train] "
                            f"epoch={epoch} step={global_step} "
                            f"loss={log_payload['loss']:.4f} "
                            f"mel={log_payload['mel']:.4f} "
                            f"wav_l1={log_payload['wav_l1']:.4f} "
                            f"feat_l1={log_payload['feat_l1']:.4f} "
                            f"stft={log_payload['stft']:.4f} "
                            f"vq={log_payload['vq']:.4f} "
                            f"gan_mp={log_payload['gan_mp']:.4f} "
                            f"gan_mrd={log_payload['gan_mrd']:.4f} "
                            f"fm_mp={log_payload['fm_mp']:.4f} "
                            f"fm_mrd={log_payload['fm_mrd']:.4f} "
                            f"disc={log_payload['disc']:.4f} "
                            f"ppl={log_payload['ppl']:.2f} "
                            f"active={log_payload['active']:.2f} "
                            f"spk_ce={log_payload['spk_ce']:.4f} "
                            f"lr={current_lr:.7f}"
                        )

                if global_step % save_every == 0:
                    save_checkpoint(
                        output_dir / "checkpoints" / f"step_{global_step:08d}.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        global_step,
                        best_val_loss,
                        cfg,
                        discriminator_state=_discriminator_state(
                            mpd,
                            mrd,
                            optimizer_d,
                            scheduler_d,
                            scaler_d,
                        ),
                    )

                if valid_loader is not None and global_step % eval_every == 0:
                    if is_dist_initialized():
                        dist.barrier()

                    metrics = evaluate(
                        model,
                        frontend,
                        valid_loader,
                        mel_loss_fn,
                        stft_loss_fn,
                        cfg,
                        device,
                    )

                    if is_main_process():
                        print(
                            "[valid] "
                            f"step={global_step} "
                            f"loss={metrics['total']:.4f} "
                            f"mel={metrics['mel']:.4f} "
                            f"wav_l1={metrics['waveform_l1']:.4f} "
                            f"feat_l1={metrics['feature_l1']:.4f} "
                            f"stft={metrics['stft']:.4f} "
                            f"vq={metrics['vq']:.4f}"
                        )

                    val_loss = metrics["total"]
                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint(
                            output_dir / "checkpoints" / "best.pt",
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            epoch,
                            global_step,
                            best_val_loss,
                            cfg,
                            discriminator_state=_discriminator_state(
                                mpd,
                                mrd,
                                optimizer_d,
                                scheduler_d,
                                scaler_d,
                            ),
                        )

                if global_step >= max_steps:
                    save_checkpoint(
                        output_dir / "checkpoints" / "last.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        global_step,
                        best_val_loss,
                        cfg,
                        discriminator_state=_discriminator_state(
                            mpd,
                            mrd,
                            optimizer_d,
                            scheduler_d,
                            scaler_d,
                        ),
                    )
                    if is_main_process():
                        print(f"[done] reached max_steps={max_steps}")
                    return

            save_checkpoint(
                output_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                global_step,
                best_val_loss,
                cfg,
                discriminator_state=_discriminator_state(
                    mpd,
                    mrd,
                    optimizer_d,
                    scheduler_d,
                    scaler_d,
                ),
            )

        save_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            max_epochs,
            global_step,
            best_val_loss,
            cfg,
            discriminator_state=_discriminator_state(
                mpd,
                mrd,
                optimizer_d,
                scheduler_d,
                scaler_d,
            ),
        )
    finally:
        cleanup_ddp()


__all__ = ["evaluate", "train_bicodec"]
