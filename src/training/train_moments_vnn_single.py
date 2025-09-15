import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pyarrow.parquet as pq
import os
from src.networks.vnn.moments_vnn import VNN, general_distribution_loss, l2_distribution_loss
from config.config import settings


def load_data_from_parquet(parquet_path):
    """
    Load analytical moments and ground truth S2 distribution from parquet file.
    
    Args:
        parquet_path: Path to the parquet file containing analytical moments
    
    Returns:
        M1: First moment tensor
        M2: Second moment tensor  
        target_points: Ground truth S2 points in 3D coordinates
        target_weights: Ground truth weights
    """
    # Read the parquet file
    table = pq.read_table(parquet_path)
    data = table.to_pandas()
    
    # Extract first moment
    first_moment_flat = np.array(data['first_moment'].iloc[0])
    first_moment_shape = data['first_moment_shape'].iloc[0]
    M1 = torch.tensor(first_moment_flat.reshape(first_moment_shape), dtype=torch.float32)
    
    # Extract second moment
    second_moment_flat = np.array(data['second_moment'].iloc[0])
    second_moment_shape = data['second_moment_shape'].iloc[0]
    M2 = torch.tensor(second_moment_flat.reshape(second_moment_shape), dtype=torch.float32)
    
    # Determine distribution type and load metadata accordingly
    distribution_type = data['distribution_type'].iloc[0]
    if distribution_type == 's2_delta_mixture':
        weights = data['weights'].iloc[0]
        s2_points = data['s2_points'].iloc[0]
        
        # Convert to tensors directly from the extracted data
        weights = torch.tensor(weights, dtype=torch.float32)
        s2_points = torch.tensor(s2_points, dtype=torch.float32)
        
        distribution_metadata = {
            'type': 's2_delta_mixture',
            'weights': weights.numpy(),
            's2_points': s2_points.numpy()
        }
        target = {'weights': weights,
                  's2_points': s2_points}
    elif distribution_type == 'vmf_mixture':
        means = data['means'].iloc[0]
        kappas = data['kappas'].iloc[0]
        weights = data['weights'].iloc[0]

        print(means, kappas, weights)
        # Convert to tensors, handling means as array of arrays
        means = torch.tensor(np.stack(means), dtype=torch.float32)
        kappas = torch.tensor(kappas, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)
        
        distribution_metadata = {
            'type': 'vmf_mixture',
            'means': means.numpy(),
            'kappas': kappas.numpy(),
            'weights': weights.numpy()
        }
        target = {'means': means,
                  'kappas': kappas,
                  'weights': weights}
    else:
        raise ValueError(f"Unrecognized or missing distribution_type in parquet file: {distribution_type}")

    return M1, M2, distribution_metadata, distribution_type, target


def train_model(model, M1, M2, target, distribution_type, num_epochs, lr, device):
    """
    Train the VNN model on a single example.

    Args:
        model: VNN model to train
        M1: First moment tensor (L, L)
        M2: Second moment tensor (L, L, L, L)
        target: Dictionary of target tensors
        distribution_type: String specifying distribution type
        num_epochs: Number of training epochs
        lr: Learning rate
        device: torch.device object (e.g. 'cuda:0' or 'cpu')
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Reshape moments for VNN input
    # M1 should be (batch_size, N, 1) where N is the flattened dimension
    # M2 should be (batch_size, N, N) as a covariance matrix
    
    L = M1.shape[0]  # Image resolution
    
    # Flatten M1 and add batch and channel dimensions, then move to device
    M1_flat = M1.flatten()  # (L*L,)
    M1_batch = M1_flat.unsqueeze(0).unsqueeze(-1).to(device)  # (1, L*L, 1)
    
    # Reshape M2 from (L, L, L, L) to (L*L, L*L) covariance matrix
    M2_reshaped = M2.view(L*L, L*L)  # (L*L, L*L)
    M2_batch = M2_reshaped.unsqueeze(0).to(device)  # (1, L*L, L*L)

    # Prepare batch for all target fields
    target_batch = {k: v.unsqueeze(0).to(device) for k, v in target.items()}
    print(f"Training shapes:")
    print(f"M1: {M1_batch.shape}")
    print(f"M2: {M2_batch.shape}")
    for k, v in target_batch.items():
        print(f"Target {k}: {v.shape}")
    
    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Forward pass
        pred = model(M2_batch, M1_batch)
        # Compute loss using L2 loss function
        loss = l2_distribution_loss(pred, target_batch, distribution_type)
        loss.backward()
        optimizer.step()
        if epoch % settings.logging.print_interval == 0:
            print(f"Epoch {epoch:4d}, Loss: {loss.item():.6f}")
            if settings.logging.verbose:
                with torch.no_grad():
                    for k in pred:
                        print(f"  Predicted {k}: {pred[k][0].detach().cpu().numpy()}")
                    for k in target_batch:
                        print(f"  Target {k}:    {target_batch[k][0].cpu().numpy()}")
    return model


if __name__ == "__main__":
    device = torch.device(
        f"cuda:{settings.device.cuda_device}" 
        if settings.device.use_cuda and torch.cuda.is_available() 
        else "cpu"
    )
    print(f"Using device: {device}")
    # Load data from parquet file
    print(f"Loading data from {settings.data_generation.parquet_path}...")
    M1, M2, distribution_metadata, distribution_type, target = load_data_from_parquet(settings.data_generation.parquet_path)
    print(f"Distribution type: {distribution_type}")
    model = VNN(degree=settings.model.architecture.vnn_layer_degree,
                hidden_dim=settings.model.architecture.hidden_dim,
                distribution_type=distribution_type,
                distribution_metadata=distribution_metadata).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    # Load model if exists
    if os.path.exists(settings.model.checkpoint.load_path):
        model.load_state_dict(torch.load(settings.model.checkpoint.load_path))
        print(f"Loaded model parameters from: {settings.model.checkpoint.load_path}")
    else:
        print(f"No saved model found. Training from scratch.")
    # Train the model
    print("Starting training...")
    trained_model = train_model(model, M1, M2, target, distribution_type,
                               num_epochs=settings.training.num_epochs,
                               lr=settings.training.learning_rate, device=device)
    # Save the trained model parameters
    torch.save(trained_model.state_dict(), settings.model.checkpoint.save_path)
    print(f"Model parameters saved to: {settings.model.checkpoint.save_path}")
    # Final evaluation
    print("\nFinal evaluation:")
    trained_model.eval()
    with torch.no_grad():
        L = M1.shape[0]
        M1_flat = M1.flatten().unsqueeze(0).unsqueeze(-1).to(device)
        M2_reshaped = M2.view(L*L, L*L).unsqueeze(0).to(device)
        pred = trained_model(M2_reshaped, M1_flat)
        
        # Prepare target batch for final evaluation
        target_batch = {k: v.unsqueeze(0).to(device) for k, v in target.items()}
        
        final_loss = l2_distribution_loss(pred, target_batch, distribution_type)
        print(f"Final loss: {final_loss.item():.6f}")
        for k in pred:
            print(f"Final predicted {k}: {pred[k][0].cpu().numpy()}")
        for k in target_batch:
            print(f"Target {k}: {target_batch[k][0].cpu().numpy()}")
