import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def finetune_encoder_wrapper(args, encoder, train_loader, test_loader=None):
    """
    Fine-tune the encoder using TripletMarginLoss with hard negative mining.
    Only the last 50% of the layers are trained.
    Includes stabilizing strategies to promote decreasing loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetune_lr = 1e-4
    triplet_margin = 1.2
    finetune_epochs = 10
    gradient_clip_norm = 1.0

    encoder = encoder.to(device)

    # Freeze first 50% of layers
    all_params = list(encoder.named_parameters())
    total = len(all_params)
    freeze_count = int(total * 0.5)
    for i, (name, param) in enumerate(all_params):
        param.requires_grad = i >= freeze_count
    print(f"[FT] Unfrozen {total - freeze_count}/{total} params")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, encoder.parameters()),
        lr=finetune_lr
    )

    # Scheduler (linear warmup then step decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # Triplet loss
    triplet_criterion = nn.TripletMarginLoss(margin=triplet_margin, p=2, reduction='mean')

    encoder.train()
    for epoch in range(finetune_epochs):
        total_loss = 0.0
        valid_batches = 0

        for batch in tqdm(train_loader, desc=f"FT Epoch {epoch+1}/{finetune_epochs}"):
            images, _, masks = batch[:3]
            images = images.to(device)
            masks = masks.to(device)

            features = encoder(images)[-1]  # last feature map
            bs, dim, h, w = features.size()
            features_flat = features.permute(0, 2, 3, 1).reshape(-1, dim)
            masks_flat = F.interpolate(masks, size=(h, w), mode='nearest').squeeze(1).reshape(-1)

            normal_feats = features_flat[masks_flat == 0]
            anomaly_feats = features_flat[masks_flat == 1]

            if len(normal_feats) < 4 or len(anomaly_feats) < 2:
                continue  # skip poorly informative batches

            # Normalize
            normal_feats = F.normalize(normal_feats, dim=1)
            anomaly_feats = F.normalize(anomaly_feats, dim=1)

            idx = torch.randperm(len(normal_feats))[:4]
            anchor = normal_feats[idx[0]].unsqueeze(0)
            positive = normal_feats[idx[1]].unsqueeze(0)
            semi_hard_positive = normal_feats[idx[2]].unsqueeze(0)
            dists = torch.cdist(anchor, anomaly_feats, p=2).squeeze(0)
            negative = anomaly_feats[torch.argmin(dists)].unsqueeze(0)

            loss = triplet_criterion(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), gradient_clip_norm)
            optimizer.step()

            total_loss += loss.item()
            valid_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(valid_batches, 1)
        print(f"[FT] Epoch {epoch+1} Loss: {avg_loss:.4f}")

        if test_loader and (epoch + 1) % 5 == 0:
            encoder.eval()
            with torch.no_grad():
                print("[FT] Optional validation skipped for simplicity.")
            encoder.train()

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = True

    return encoder
