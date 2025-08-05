import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from config.config import settings
from src.networks.moments_vnn import VNN, general_distribution_loss, l2_distribution_loss
from src.data.load_training_data import load_data_from_zarr, MomentsDataset

def train_model(model, M1_list, M2_list, target_list, distribution_type, num_epochs, lr, device):
    """
    Train the VNN model on a batch of multiple VDM examples.

    Args:
        model: VNN model to train
        M1_list (list[Tensor]): List of first moment tensors
        M2_list (list[Tensor]): List of second moment tensors
        target_list (list[dict]): List of target dicts for loss computation
        distribution_type (str): Distribution type string
        num_epochs (int): Number of training epochs
        lr (float): Learning rate
        device: torch.device object (e.g. 'cuda:0' or 'cpu')

    Returns:
        model: The trained VNN model
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    batch_size = settings.training.batch_size
    L = M1_list[0].shape[0]

    dataset = MomentsDataset(M1_list, M2_list, target_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_data = train_model.val_data
    
    # Setup blur scheduling from settings
    use_blur_scheduling = settings.loss.sinkhorn.use_blur_scheduling
    if use_blur_scheduling:
        start_blur, end_blur = settings.loss.sinkhorn.blur_schedule
        print(f"Using exponential blur scheduling: {start_blur} -> {end_blur} over {num_epochs} epochs")
    else:
        print(f"Using fixed blur parameter: {settings.loss.sinkhorn.blur}")
    
    if not settings.logging.verbose:
        epoch_iter = range(num_epochs)
    else:
        from tqdm import tqdm
        epoch_iter = tqdm(range(num_epochs), desc='Epochs')
            
    for epoch in epoch_iter:
        # Calculate current blur parameter using exponential scheduling
        if use_blur_scheduling:
            # Exponential decay: blur = start_blur * (end_blur/start_blur)^(epoch/(num_epochs-1))
            progress = epoch / max(1, num_epochs - 1)  # Progress from 0 to 1
            current_blur = start_blur * (end_blur / start_blur) ** progress
        else:
            current_blur = None  # Use default from settings
        
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            M1_batch_list, M2_batch_list, target_batch_list = batch
            optimizer.zero_grad()
            M1_batch = torch.stack([m.flatten().unsqueeze(-1) for m in M1_batch_list]).to(device)
            M2_batch = torch.stack([m.view(L*L, L*L) for m in M2_batch_list]).to(device)
            
            # target_batch_list is already a dictionary with stacked tensors
            target_batch = {k: v.to(device) for k, v in target_batch_list.items()}
            pred = model(M2_batch, M1_batch)
            loss = l2_distribution_loss(pred, target_batch, distribution_type)
            loss_scalar = loss.mean()  # shape: ()
            loss_scalar.backward()
            epoch_loss += loss_scalar.item()
            optimizer.step()
            num_batches += 1
        avg_loss = epoch_loss / max(1, num_batches)
        if epoch % settings.logging.print_interval == 0:
            blur_info = f", Blur: {current_blur:.4f}" if use_blur_scheduling else ""
            print(f"Epoch {epoch:4d}, Loss: {avg_loss:.6f}{blur_info}")
            # Validation loss if val_data is set
            if val_data is not None:
                M1_val, M2_val, target_val, dist_type_val = val_data
                with torch.no_grad():
                    pred_val = model(M2_val, M1_val)
                    val_keys = [k for k in target_val[0].keys()]
                    target_batch_val = {k: torch.stack([t[k] for t in target_val]).to(device) for k in val_keys}
                    val_loss = l2_distribution_loss(pred_val, target_batch_val, dist_type_val)
                    val_loss_scalar = val_loss.mean()  # shape: ()
                    print(f"  Validation loss: {val_loss_scalar.item():.6f}") 

            if settings.logging.verbose:
                with torch.no_grad():
                    print(f"  --- Training Details (First Example) ---")
                    print(f"  Distribution type: {distribution_type}")
                    print(f"  Batch size: {pred['weights'].shape[0]}")
                    for k in pred:
                        print(f"  Training Predicted {k}: {pred[k][0].detach().cpu().numpy()}")
                    for k in target_batch:
                        print(f"  Training Target {k}:    {target_batch[k][0].cpu().numpy()}")
                        
                    print(f"  --- Validation Details (First Example) ---")
                    print(f"  Distribution type: {dist_type_val}")
                    for k in pred_val:
                        print(f"  Validation Predicted {k}: {pred_val[k][0].detach().cpu().numpy()}")
                    for k in target_batch_val:
                        print(f"  Validation Target {k}:    {target_batch_val[k][0].cpu().numpy()}")
    return model

if __name__ == "__main__":
    device = torch.device(
        f"cuda:{settings.device.cuda_device}"
        if settings.device.use_cuda and torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load train and validation data from separate Zarr files
    zarr_train_path = os.path.join(settings.data_generation.zarr.save_dir, "train_vdms.zarr")
    zarr_val_path = os.path.join(settings.data_generation.zarr.save_dir, "val_vdms.zarr")
    print(f"Loading training data from {zarr_train_path}...")
    M1_train, M2_train, distribution_meta_train_list, dist_type = load_data_from_zarr(zarr_train_path)
    print(f"Loaded {len(M1_train)} training examples from Zarr.")
    print(f"Loading validation data from {zarr_val_path}...")
    M1_val, M2_val, distribution_meta_val_list, _ = load_data_from_zarr(zarr_val_path)
    print(f"Loaded {len(M1_val)} validation examples from Zarr.")

    distribution_meta_train = distribution_meta_train_list[0] if len(distribution_meta_train_list) > 0 else None

    # Model construction
    model = VNN(
        degree=settings.model.architecture.vnn_layer_degree,
        hidden_dim=settings.model.architecture.hidden_dim,
        distribution_type=dist_type,
        distribution_metadata=distribution_meta_train
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Load model if exists
    if os.path.exists(settings.model.checkpoint.load_path):
        model.load_state_dict(torch.load(settings.model.checkpoint.load_path, weights_only=True))
        print(f"Loaded model parameters from: {settings.model.checkpoint.load_path}")
    else:
        print(f"No saved model found. Training from scratch.")

    # Prepare validation data for logging during training

    L_val = M1_val[0].shape[0]
    M1_val_batch = torch.stack([m.flatten().unsqueeze(-1) for m in M1_val]).to(device)
    M2_val_batch = torch.stack([m.view(L_val*L_val, L_val*L_val) for m in M2_val]).to(device)
    val_data = (M1_val_batch, M2_val_batch, distribution_meta_val_list, dist_type)
    train_model.val_data = val_data

    # Train the model
    print(f"Starting training on {len(M1_train)} examples...")
    trained_model = train_model(
        model, M1_train, M2_train, distribution_meta_train_list, dist_type,
        num_epochs=settings.training.num_epochs,
        lr=settings.training.learning_rate, device=device
    )

    # Save the trained model parameters
    torch.save(trained_model.state_dict(), settings.model.checkpoint.save_path)
    print(f"Model parameters saved to: {settings.model.checkpoint.save_path}")

    # Final validation
    print("\nValidation:")
    trained_model.eval()
    
    # Get final blur value for validation from settings
    use_blur_scheduling = settings.loss.sinkhorn.use_blur_scheduling
    final_blur = None
    if use_blur_scheduling:
        blur_schedule = settings.loss.sinkhorn.blur_schedule
        final_blur = blur_schedule[1]  # end_blur
    
    with torch.no_grad():
        if len(M1_val) > 0:
            L_val = M1_val[0].shape[0]
            M1_val_batch = torch.stack([m.flatten().unsqueeze(-1) for m in M1_val]).to(device)
            M2_val_batch = torch.stack([m.view(L_val*L_val, L_val*L_val) for m in M2_val]).to(device)
            pred = trained_model(M2_val_batch, M1_val_batch)
            val_keys = [k for k in distribution_meta_val_list[0].keys()]
            target_batch = {k: torch.stack([t[k] for t in distribution_meta_val_list]).to(device) for k in val_keys}
            val_loss = l2_distribution_loss(pred, target_batch, dist_type)
            if isinstance(val_loss, tuple):
                val_loss_val = val_loss[0] if isinstance(val_loss[0], torch.Tensor) else val_loss[0]
                print(f"Validation loss: {val_loss_val.item():.6f}")
            else:
                print(f"Validation loss: {val_loss.item():.6f}")
            for k in pred:
                print(f"Validation predicted {k}: {pred[k][0].cpu().numpy()}")
            for k in target_batch:
                print(f"Validation target {k}: {target_batch[k][0].cpu().numpy()}")
        else:
            print("No validation examples available.")
