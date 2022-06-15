from collections import namedtuple
import jax
import jax.numpy as jnp
import numpy as np
import wandb

def un_jax_log_f(x):
    if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
        if len(x.shape) == 0:
            return x.item()
        else:
            return x.tolist()
    return x

def un_jax_logs(logs):
    return jax.tree_util.tree_map(un_jax_log_f, logs)

LogTuple = namedtuple('LogTuple', ['mean', 'count'])

def is_scalar(x):
    return isinstance(x, int) or isinstance(x, float) or (isinstance(x, jnp.ndarray) and len(x.shape) == 0) or (isinstance(x, np.ndarray) and len(x.shape) == 0)

def is_vector(x):
    return (isinstance(x, jnp.ndarray) and len(x.shape) > 0) or (isinstance(x, np.ndarray) and len(x.shape) > 0)

class CombineLogs:
    def __init__(self, use_wandb=False):
        self.curr_logs = None
        self.use_wandb = use_wandb
    
    @staticmethod
    def combine_elements(a, b):
        if is_scalar(a):
            a = LogTuple(a, 1,)
        if is_scalar(b):
            b = LogTuple(b, 1,)
        if isinstance(a, LogTuple) and isinstance(b, LogTuple):
            if (a.count + b.count) == 0:
                return LogTuple(0.0, 0)
            return LogTuple((a.mean * a.count + b.mean * b.count) / (a.count + b.count), a.count + b.count)
        if is_vector(a) and is_vector(b):
            return jnp.concatenate((a, b,), axis=0)
        raise NotImplementedError
    
    @staticmethod
    def reduce_elements(x):
        if isinstance(x, LogTuple):
            return x.mean
        if is_vector(x):
            return x.mean()
        if is_scalar(x):
            return x
        raise NotImplementedError
     
    @staticmethod
    def is_leaf(x):
        return is_vector(x) or is_scalar(x) or isinstance(x, LogTuple)
    
    def log(self, *postproc_funcs):
        logs = self.gather_logs(*postproc_funcs)
        logs = un_jax_logs(logs)
        if self.use_wandb:
            wandb.log(logs)
        print(logs)
        return logs

    def accum_logs(self, logs):
        if self.curr_logs is None:
            self.curr_logs = logs
        else:
            self.curr_logs = jax.tree_util.tree_map(self.combine_elements, self.curr_logs, logs, is_leaf=self.is_leaf)
    
    def gather_logs(self, *postproc_funcs):
        logs = jax.tree_util.tree_map(self.reduce_elements, self.curr_logs, is_leaf=self.is_leaf)
        for f in postproc_funcs:
            result = f(logs)
            if result is not None:
                logs = result
        return logs
    
    def raw_logs(self):
        return self.curr_logs

    def reset_logs(self):
        self.curr_logs = None

def label_logs(logs, label, to_add):
    return {label: logs, **to_add}
