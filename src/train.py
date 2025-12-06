import torch

def train(model, train_loader, device, epochs = 10):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.001,
        momentum=0.9,
        weight_decay=1e-4,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            n_batches += 1

        print(f"Epoch {epoch}/{epochs} | train_loss = {total_loss / n_batches:.4f}")

    torch.save(model.state_dict(), "rgbd_fasterrcnn_crossmodal_8x8.pth")
    print("Model saved: rgbd_fasterrcnn_crossmodal_8x8.pth")
