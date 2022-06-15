from functools import partial
from typing import Optional, Dict, Any
from micro_config import ConfigScript, MetaConfig
from dataclasses import dataclass, asdict
from torch.utils.data.dataset import IterableDataset
from torch.utils.data import DataLoader
from collections import deque
import jax
import os
import json
import pickle as pkl
import optax
import chex
from logs import CombineLogs, label_logs
from tqdm.auto import tqdm
import wandb

@dataclass
class TrainLoop(ConfigScript):
    model: ConfigScript
    train_data: ConfigScript
    eval_data: ConfigScript
    optim: ConfigScript
    evaluator: Optional[ConfigScript]
    rng: ConfigScript
    save_dir: Optional[str]
    max_checkpoints: Optional[int]
    epochs: int
    max_steps: Optional[int]
    bsize: int
    eval_bsize: int
    eval_batches: Optional[int]
    log_every: int
    eval_every: int
    save_every: Optional[int]
    dataloader_workers: int
    jit: bool
    use_wandb: bool
    wandb_project: str
    loss_kwargs: Dict[str, Any]

    def unroll(self, metaconfig: MetaConfig):
        print('using config:', asdict(self))
        
        # save configs
        save_dir = metaconfig.convert_path(self.save_dir)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, 'config.json'), 'w') as f:
                json.dump(asdict(self), f)
            with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
                pkl.dump(self, f)
        
        # initalize wandb
        if self.use_wandb:
            wandb.init(project=self.wandb_project, config=asdict(self))
        
        # conditionally block jit
        if not self.jit:
            fake_jit = chex.fake_jit()
            fake_jit.start()
        
        # setup dataloaders
        train_dataset = self.train_data.unroll(metaconfig)
        eval_dataset = self.eval_data.unroll(metaconfig)
        train_data_loader_kwargs = {'num_workers': self.dataloader_workers, 
                                    'batch_size': self.bsize, 
                                    'collate_fn': train_dataset.collate}
        eval_data_loader_kwargs = {'num_workers': self.dataloader_workers, 
                                   'batch_size': self.eval_bsize, 
                                   'collate_fn': eval_dataset.collate}
        if not isinstance(train_dataset, IterableDataset):
            train_data_loader_kwargs['shuffle'] = True
        if not isinstance(train_dataset, IterableDataset):
            eval_data_loader_kwargs['shuffle'] = True
        train_dataloader = DataLoader(train_dataset, **train_data_loader_kwargs)
        eval_dataloader = DataLoader(eval_dataset, **eval_data_loader_kwargs)

        # setup training objects
        evaluator, eval_state, eval_can_jit = None, None, False
        if self.evaluator is not None:
            evaluator, eval_state, eval_can_jit = self.evaluator.unroll(metaconfig)
        if eval_can_jit:
            evaluator = jax.jit(evaluator, static_argnums=0)
        model, params, model_state = self.model.unroll(metaconfig)
        optim, opt_state = self.optim.unroll(metaconfig)

        # define training step
        @partial(jax.jit, static_argnames=list(self.loss_kwargs.keys()))
        def step_fn(params, model_state, opt_state, rng, *args, **kwargs):
            def grad_loss(*args, **kwargs):
                (loss, s1), s2 = model.loss(*args, **kwargs)
                return loss, (s1, s2)
            
            (_, ((logs, postproc_fs,), model_state,),), grads = jax.value_and_grad(grad_loss, has_aux=True)(params, model_state, rng, *args, is_training=True, **kwargs)
            updates, opt_state = optim(grads, opt_state, params=params)
            params = optax.apply_updates(params, updates)
            return (logs, postproc_fs,), params, model_state, opt_state
        
        # define eval loss
        @partial(jax.jit, static_argnames=list(self.loss_kwargs.keys()))
        def eval_loss(params, model_state, rng, *args, **kwargs):
            (loss, (logs, postproc_fs,),), model_state = model.loss(params, model_state, rng, *args, is_training=False, **kwargs)
            return (loss, (logs, postproc_fs,),), model_state

        # setup logs
        train_logs = CombineLogs(use_wandb=self.use_wandb)
        eval_logs = CombineLogs(use_wandb=self.use_wandb)
        eval_loss_accum = CombineLogs(use_wandb=False)

        # initalize training loop state
        step = 0
        best_loss = float('inf')
        saved_checkpoints = deque([])
        rng = self.rng.unroll(metaconfig)

        # train loop
        for epoch in tqdm(range(self.epochs)):
            for items in tqdm(train_dataloader):
                
                # step model and accumulate training logs
                rng, new_rng = jax.random.split(rng)
                (logs, postproc_fs,), params, model_state, opt_state = step_fn(params, model_state, opt_state, new_rng, *items, **self.loss_kwargs)
                train_logs.accum_logs(logs)
                
                # publish training logs
                if (step + 1) % self.log_every == 0:
                    train_logs.log(*postproc_fs, 
                                   partial(label_logs, label='train', to_add={'step': step, 'epoch': epoch}))
                
                # clear training logs
                if (step + 1) % self.optim.grad_accum_steps == 0:
                    train_logs.reset_logs()
                
                # begin evaluation
                if (step + 1) % self.eval_every == 0:
                    
                    # clear eval logs
                    eval_logs.reset_logs()
                    eval_loss_accum.reset_logs()
                    
                    # eval on batches
                    for i, eval_items in enumerate(eval_dataloader):
                        
                        # conditionally terminate early
                        if self.eval_batches is not None and i >= self.eval_batches:
                            break
                        
                        # get eval logs
                        rng, new_rng = jax.random.split(rng)
                        (loss, (logs, postproc_fs,),), model_state = eval_loss(params, model_state, new_rng, *eval_items, **self.loss_kwargs)
                        
                        # conditionally run evaluator
                        if self.evaluator is not None:
                            rng, new_rng = jax.random.split(rng)
                            evaluator_logs, model_state, eval_state = evaluator(model, params, model_state, eval_state, new_rng, eval_items)
                            
                            # conditionally fold evaluation results into logs
                            if evaluator_logs is not None:
                                logs['evaluation'] = evaluator_logs
                        
                        # accumulate eval logs
                        eval_logs.accum_logs(logs)
                        eval_loss_accum.accum_logs(loss)
                    
                    # publish eval logs
                    eval_logs.log(*postproc_fs, 
                                  partial(label_logs, label='eval', to_add={'step': step, 'epoch': epoch}))
                    
                    # conditionally save best model and optimizer state
                    loss = eval_loss_accum.gather_logs().item()
                    if save_dir is not None and loss < best_loss:
                        print('new best eval loss! Saving ...')
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        with open(os.path.join(save_dir, 'model.pkl'), 'wb') as f:
                            pkl.dump((params, model_state,), f)
                        with open(os.path.join(save_dir, 'optim.pkl'), 'wb') as f:
                            pkl.dump(opt_state, f)
                        print('saved.')
                        best_loss = loss
                
                # periodically save checkpoint
                if save_dir is not None and self.save_every is not None and (step + 1) % self.save_every == 0:
                    print('saving checkpoint...')
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    # conditionally delete old checkpoints
                    if (self.max_checkpoints is not None) and (len(saved_checkpoints) >= self.max_checkpoints):
                        os.system('rm -rf %s' % (saved_checkpoints.popleft()))
                    
                    # save
                    with open(os.path.join(save_dir, 'model_%d.pkl' % (step)), 'wb') as f:
                        pkl.dump((params, model_state,), f)
                    saved_checkpoints.append(os.path.join(save_dir, 'model_%d.pkl' % (step)))
                    print('saved.')
                
                # increment step counter
                step += 1
                
                # conditionally terminate
                if self.max_steps is not None and step >= self.max_steps:
                    return
        
        # undo conditional jit block
        if not self.jit:
            fake_jit.stop()
                        
