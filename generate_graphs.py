import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import os

# Create directory for images
OUTPUT_DIR = "scheduler_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def plot_schedule(scheduler_cls, base_params, variations, name, epochs=100, metric_values=None):
    # --- 1. Base Plot Only ---
    plt.figure(figsize=(12, 4))
    
    model = torch.nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = scheduler_cls(optimizer, **base_params)
    
    lrs = []
    for epoch in range(epochs):
        lrs.append(get_lr(optimizer))
        optimizer.step()
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
             val = metric_values[epoch] if metric_values else 0.5
             scheduler.step(val)
        else:
             scheduler.step()
    
    plt.plot(lrs, label=f"Base: {base_params}", linewidth=4)
    plt.title(f"{name} (Base Parameter)")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    base_filename = os.path.join(OUTPUT_DIR, f"{name}_base.png")
    plt.savefig(base_filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved {base_filename}")
    
    # Store base LRs for the variation plot to avoid re-running base simulation if desired, 
    # but re-running is safer/simpler structure-wise.
    base_lrs = lrs 

    # --- 2. Variation Plot (Base + Variations) ---
    plt.figure(figsize=(12, 4))
    
    # Re-plot base
    plt.plot(base_lrs, label=f"Base: {base_params}", linewidth=4)

    # Variations
    for param_name, new_value in variations.items():
        model = torch.nn.Linear(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        # Merge base params with the variation
        current_params = base_params.copy()
        current_params[param_name] = new_value
        
        try:
            scheduler = scheduler_cls(optimizer, **current_params)
            
            lrs_var = []
            for epoch in range(epochs):
                lrs_var.append(get_lr(optimizer))
                optimizer.step()
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    val = metric_values[epoch] if metric_values else 0.5
                    scheduler.step(val)
                else:
                    scheduler.step()
            
            plt.plot(lrs_var, linestyle='--', label=f"{param_name}={new_value}", linewidth=3)
        except Exception as e:
            print(f"Skipping variation {param_name}={new_value} for {name} due to error: {e}")

    plt.title(f"{name} (Parameter Variations)")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    var_filename = os.path.join(OUTPUT_DIR, f"{name}_variation.png")
    plt.savefig(var_filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved {var_filename}")

def main():
    # --- StepLR ---
    plot_schedule(
        lr_scheduler.StepLR,
        base_params={'step_size': 30, 'gamma': 0.1},
        variations={'step_size': 10, 'gamma': 0.5},
        name="StepLR"
    )

    # --- MultiStepLR ---
    plot_schedule(
        lr_scheduler.MultiStepLR,
        base_params={'milestones': [30, 80], 'gamma': 0.1},
        variations={'milestones': [20, 60], 'gamma': 0.5},
        name="MultiStepLR"
    )

    # --- ConstantLR ---
    plot_schedule(
        lr_scheduler.ConstantLR,
        base_params={'factor': 0.33, 'total_iters': 5},
        variations={'factor': 0.1, 'total_iters': 20},
        name="ConstantLR",
        epochs=50 # Shorten for clarity
    )

    # --- LinearLR ---
    plot_schedule(
        lr_scheduler.LinearLR,
        base_params={'start_factor': 0.33, 'total_iters': 5},
        variations={'start_factor': 0.1, 'total_iters': 20},
        name="LinearLR",
        epochs=50
    )

    # --- ExponentialLR ---
    plot_schedule(
        lr_scheduler.ExponentialLR,
        base_params={'gamma': 0.9},
        variations={'gamma': 0.95},
        name="ExponentialLR"
    )

    # --- PolynomialLR ---
    plot_schedule(
        lr_scheduler.PolynomialLR,
        base_params={'total_iters': 50, 'power': 1.0},
        variations={'power': 2.0, 'total_iters': 30},
        name="PolynomialLR",
        epochs=60
    )

    # --- CosineAnnealingLR ---
    plot_schedule(
        lr_scheduler.CosineAnnealingLR,
        base_params={'T_max': 50, 'eta_min': 0},
        variations={'T_max': 25, 'eta_min': 0.05},
        name="CosineAnnealingLR"
    )

    # --- ReduceLROnPlateau ---
    metrics = [1.0] * 100 # Flat loss
    plot_schedule(
        lr_scheduler.ReduceLROnPlateau,
        base_params={'mode': 'min', 'factor': 0.1, 'patience': 10},
        variations={'patience': 5, 'factor': 0.5},
        name="ReduceLROnPlateau",
        metric_values=metrics
    )

    # --- CyclicLR ---
    # Needs a custom loop usually, but step() works if called batch-wise, 
    # here we assume 1 batch per epoch or viewing it per-step
    plot_schedule(
        lr_scheduler.CyclicLR,
        base_params={'base_lr': 0.001, 'max_lr': 0.1, 'step_size_up': 5, 'mode': 'triangular'},
        variations={'step_size_up': 20, 'mode': 'triangular2'},
        name="CyclicLR",
        epochs=100
    )

    # --- OneCycleLR ---
    plot_schedule(
        lr_scheduler.OneCycleLR,
        base_params={'max_lr': 0.1, 'total_steps': 100},
        variations={'pct_start': 0.5},
        name="OneCycleLR",
        epochs=100
    )

    # --- CosineAnnealingWarmRestarts ---
    plot_schedule(
        lr_scheduler.CosineAnnealingWarmRestarts,
        base_params={'T_0': 10, 'T_mult': 1},
        variations={'T_0': 20, 'T_mult': 2},
        name="CosineAnnealingWarmRestarts"
    )

if __name__ == "__main__":
    main()
