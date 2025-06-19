import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def finetune_encoder_wrapper(args, encoder, train_loader, test_loader=None):
    """
    Fine-tune the encoder using TripletMarginLoss with stabilized hard negative mining.
    Only the last 50% of the layers are trained.
    Includes variance-reducing strategies to encourage consistent loss decrease.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    finetune_lr = 5e-4
    triplet_margin = 1.0
    finetune_epochs = 10
    gradient_clip_norm = 0.5

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
        lr=finetune_lr, weight_decay=1e-4
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-6)

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

            if len(normal_feats) < 10 or len(anomaly_feats) < 4:
                continue  # skip poorly informative batches

            # Normalize
            normal_feats = F.normalize(normal_feats, dim=1)
            anomaly_feats = F.normalize(anomaly_feats, dim=1)

            # Average multiple triplet losses per batch to reduce variance
            batch_triplets = 8
            batch_loss = 0.0
            for _ in range(batch_triplets):
                idx = torch.randperm(len(normal_feats))[:2]
                anchor = normal_feats[idx[0]].unsqueeze(0)
                positive = normal_feats[idx[1]].unsqueeze(0)
                dists = torch.cdist(anchor, anomaly_feats, p=2).squeeze(0)
                negative = anomaly_feats[torch.argmin(dists)].unsqueeze(0)
                batch_loss += triplet_criterion(anchor, positive, negative)
            batch_loss /= batch_triplets

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), gradient_clip_norm)
            optimizer.step()

            total_loss += batch_loss.item()
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
