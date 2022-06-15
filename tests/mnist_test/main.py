from base_configs import MNISTDataConfig, MLPConfig, AdamWConfig, RNGSeed, MNISTPerNumberEvaluatorConfig
from train_loop import TrainLoop
from micro_config import MetaConfig, deep_replace, parse_args
import os

seed = RNGSeed(0)

train_data = MNISTDataConfig(split='train')
eval_data = MNISTDataConfig(split='test')

model = MLPConfig(
    shapes=[28*28, 128, 128, 10], 
    dropout=0.5, 
    rng=seed.split(1), 
    checkpoint_path='outputs/mnist_test/model.pkl', 
)

evaluator = MNISTPerNumberEvaluatorConfig(
    rng=seed.split(2), 
)

optim = AdamWConfig(
    lr=3e-4, 
    weight_decay=0.00, 
    grad_accum_steps=1, 
    model=model, 
    state_path='outputs/mnist_test/optim.pkl', 
)

train = TrainLoop(
    model=model, 
    train_data=train_data, 
    eval_data=eval_data, 
    optim=optim, 
    evaluator=evaluator, 
    rng=seed.split(3), 
    save_dir='outputs/mnist_test/', 
    max_checkpoints=1, 
    epochs=10, 
    max_steps=None, 
    bsize=32, 
    eval_bsize=32, 
    eval_batches=None, 
    log_every=4096, 
    eval_every=4096, 
    save_every=None, 
    dataloader_workers=0, 
    jit=True, 
    use_wandb=False, 
    wandb_project='jax_mnist_test', 
    loss_kwargs={}, 
)

if __name__ == "__main__":
    metaconfig = MetaConfig(
        project_root=os.path.join(os.path.dirname(__file__)), 
        verbose=True, 
    )
    train = deep_replace(train, **parse_args())
    train.unroll(metaconfig)
