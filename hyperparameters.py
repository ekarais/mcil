import torch
import copy

mnist = {
    
    # SparseNet initialization
    "network_hyperparameters" : {
        "n" : 150,
        "k" : 50,
        "outChannels" : 30,
        "c_k" : 400,
        "inputSize" : (1, 28, 28),
        "outputSize" : 10,
        "kInferenceFactor" : 1.5,
        "weightSparsity" : 0.3,
        "weightSparsityCNN" : 1.0,
        "boostStrength" : 1.5,
        "boostStrengthFactor" : 0.85,
        "dropout" : 0.0,
        "useBatchNorm" : False
    },
    
    # data
    "batch_size" : 64,
    "test_batch_size" : 1000,
    "first_epoch_batch_size" : 4,
    
    # training
    "num_epochs" : 15,
    "learning_rate" : 0.01,
    "optimizer" : torch.optim.SGD,
    "lr_scheduler" : torch.optim.lr_scheduler.StepLR,
    "lr_scheduler_params" : {'step_size' : 1, 'gamma' : 0.8},
    
    # unused
    "batches_in_epoch" : 100000,
    "test_noise_every_epoch" : True,
    "momentum" : 0.0,
    "validation" : 1.0
}

cifar10 = copy.deepcopy(mnist)
cifar10["network_hyperparameters"]["inputSize"] = (3, 32, 32)

cifar100 = copy.deepcopy(cifar10)
cifar10["outputSize"] = 100