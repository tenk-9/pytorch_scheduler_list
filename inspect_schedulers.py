import torch.optim.lr_scheduler as lr_scheduler
import inspect

schedulers = [
    lr_scheduler.StepLR,
    lr_scheduler.MultiStepLR,
    lr_scheduler.ConstantLR,
    lr_scheduler.LinearLR,
    lr_scheduler.ExponentialLR,
    lr_scheduler.PolynomialLR,
    lr_scheduler.CosineAnnealingLR,
    lr_scheduler.ReduceLROnPlateau,
    lr_scheduler.CyclicLR,
    lr_scheduler.OneCycleLR,
    lr_scheduler.CosineAnnealingWarmRestarts,
]

print("Available Schedulers and their specific parameters:")
for cls in schedulers:
    sig = inspect.signature(cls.__init__)
    # Filter out self and optimizer which are common
    params = [
        f"{name}={param.default}" if param.default != inspect.Parameter.empty else name
        for name, param in sig.parameters.items()
        if name not in ['self', 'optimizer']
    ]
    print(f"{cls.__name__}: {', '.join(params)}")
