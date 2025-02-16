import os
import torch
from tqdm import tqdm
import json 


def save_checkpoint(experiment_name, model, optimizer, scheduler, epoch, step, batch_idx,
                    train_losses, train_accuracies, val_losses, val_accuracies, step_history,
                    best=False):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'step_history': step_history,  # per-step information
        'best_val_loss': min(val_losses) if val_losses else float('inf'),
    }
    suffix = "_best" if best else ""
    save_dir = os.path.join("checkpoints", experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"checkpoint_{epoch}_{step}{suffix}.pt")
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

    # Also dump the step history to JSON.
    json_path = os.path.join(save_dir, "step_history.json")
    with open(json_path, "w") as f:
        json.dump(step_history, f, indent=4)
    print(f"Step history saved to {json_path}")

def load_checkpoint(experiment_name, model, optimizer, scheduler):
    checkpoint_dir = os.path.join("checkpoints", experiment_name)
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
    if not checkpoints:
        return None
    latest = sorted(checkpoints)[-1]
    checkpoint = torch.load(os.path.join(checkpoint_dir, latest))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint

# --- Training Function ---

def train_classification(
    model,
    experiment_name,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs,
    device,
    checkpoint_frequency="epoch",  # Can be an integer (e.g., 100) or "epoch"
    resume=False
):
    # Lists to store epoch-level metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    # List to store per-step details
    step_history = []
    
    start_epoch = 0
    step = 0
    best_val_loss = float('inf')
    resume_batch = 0  # For resuming within an epoch
    
    # If resuming, load checkpoint and history.
    if resume:
        state = load_checkpoint(experiment_name, model, optimizer, scheduler)
        if state:
            start_epoch = state['epoch']
            step = state['step']
            resume_batch = state.get('batch_idx', 0)
            train_losses = state['train_losses']
            train_accuracies = state['train_accuracies']
            val_losses = state['val_losses']
            val_accuracies = state['val_accuracies']
            step_history = state.get('step_history', [])
            best_val_loss = state.get('best_val_loss', float('inf'))
            print(f"Resuming training from epoch {start_epoch}, step {step}, batch {resume_batch}")
    
    model = model.to(device)
    
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            total_samples = 0
            
            train_bar = tqdm(enumerate(train_loader), desc=f'Epoch {epoch+1}/{num_epochs} [Train]', total=len(train_loader), leave=False)
            for batch_idx, (waveforms, lengths, labels) in train_bar:
                # If resuming, skip already processed batches.
                if resume and epoch == start_epoch and batch_idx < resume_batch:
                    continue

                waveforms = waveforms.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                loss, acc, _ = model(waveforms, lengths, labels)
                loss.backward()
                optimizer.step()

                batch_size = waveforms.size(0)
                epoch_train_loss += loss.item() * batch_size
                epoch_train_acc += acc.item() * batch_size
                total_samples += batch_size

                step += 1

                # Log the step info.
                step_info = {
                    "epoch": epoch,
                    "step": step,
                    "batch_idx": batch_idx,
                    "loss": loss.item(),
                    "accuracy": acc.item()
                }
                step_history.append(step_info)
                
                train_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{acc.item():.2f}%"
                })

                # Checkpoint saving logic.
                if isinstance(checkpoint_frequency, int):
                    # Save checkpoint every 'checkpoint_frequency' steps.
                    if step % checkpoint_frequency == 0:
                        save_checkpoint(
                            experiment_name, model, optimizer, scheduler,
                            epoch, step, batch_idx, train_losses, train_accuracies,
                            val_losses, val_accuracies, step_history
                        )
                elif checkpoint_frequency == "epoch":
                    # Do not save during the batch loop.
                    pass

            avg_train_loss = epoch_train_loss / total_samples
            avg_train_acc = epoch_train_acc / total_samples
            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_acc)
            
            # --- Validation ---
            model.eval()
            epoch_val_loss = 0.0
            epoch_val_acc = 0.0
            total_val_samples = 0

            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', total=len(val_loader), leave=False)
            with torch.no_grad():
                for waveforms, lengths, labels in val_bar:
                    waveforms = waveforms.to(device)
                    lengths = lengths.to(device)
                    labels = labels.to(device)

                    loss, acc, _ = model(waveforms, lengths, labels)
                    batch_size = waveforms.size(0)
                    epoch_val_loss += loss.item() * batch_size
                    epoch_val_acc += acc.item() * batch_size
                    total_val_samples += batch_size

                    val_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{acc.item():.2f}%"
                    })
            avg_val_loss = epoch_val_loss / total_val_samples
            avg_val_acc = epoch_val_acc / total_val_samples
            val_losses.append(avg_val_loss)
            val_accuracies.append(avg_val_acc)

            # Step scheduler if using one that depends on validation loss.
            if scheduler is not None:
                scheduler.step(avg_val_loss)
            
            # Save best model checkpoint if validation loss improves.
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(
                    experiment_name, model, optimizer, scheduler,
                    epoch, step, batch_idx, train_losses, train_accuracies,
                    val_losses, val_accuracies, step_history, best=True
                )
                print(f"New best model saved with val loss: {best_val_loss:.4f} and val acc: {avg_val_acc:.2f}%")
            
            # Save checkpoint at the end of the epoch if that's the selected frequency.
            if checkpoint_frequency == "epoch":
                save_checkpoint(
                    experiment_name, model, optimizer, scheduler,
                    epoch, step, 0, train_losses, train_accuracies,
                    val_losses, val_accuracies, step_history
                )

            print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | ' 
                  f'Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%')
            
            # After resuming, disable skipping.
            if resume and epoch == start_epoch:
                resume = False
                
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current state...")
        save_checkpoint(
            experiment_name, model, optimizer, scheduler,
            epoch, step, batch_idx, train_losses, train_accuracies,
            val_losses, val_accuracies, step_history
        )
        print("Checkpoint saved. You can resume later.")
    
    return train_losses, train_accuracies, val_losses, val_accuracies, step_history


def run_training_classification(
    model,
    train_dataset,
    val_dataset,
    optimizer_class,
    optimizer_params,
    scheduler_class,
    scheduler_params,
    num_epochs,
    device,
    batch_size=64,
    num_workers=4,
    resume_training=False,
    experiment_name="classification_experiment",
    checkpoint_frequency=None  # None means save once per epoch
):
    from torch.utils.data import DataLoader
    import gc
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # Initialize optimizer and scheduler
    optimizer = optimizer_class(model.parameters(), **optimizer_params)
    scheduler = scheduler_class(optimizer, **scheduler_params) if scheduler_class is not None else None
    
    # Optionally resume from a checkpoint.
    if resume_training:
        _ = load_checkpoint(experiment_name, model, optimizer, scheduler)
    
    # Move model to device
    model.to(device)
    
    # Determine checkpoint frequency: if None, we use "epoch"
    freq = checkpoint_frequency if checkpoint_frequency is not None else "epoch"
    
    # Run the training loop; note that our train_classification now returns step_history as well.
    train_losses, train_accuracies, val_losses, val_accuracies, step_history = train_classification(
        model,
        experiment_name,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        num_epochs,
        device,
        checkpoint_frequency=freq,
        resume=resume_training
    )
    
    # Cleanup
    del train_loader, val_loader, optimizer, scheduler, model
    gc.collect()
    torch.cuda.empty_cache()
    
    return train_losses, train_accuracies, val_losses, val_accuracies, step_history
