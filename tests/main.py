from mnist_test.logs import CombineLogs, LogTuple
import numpy as np
import jax.numpy as jnp
import jax
from chex import fake_jit

if __name__ == "__main__":
    with fake_jit():
        logger = CombineLogs(use_wandb=False)
        logger.reset_logs()
        logger.accum_logs({'a': 1, 'b': 2, 'c': [1, 2, 3], 'd': LogTuple(5.5, 3), 'e': jnp.array([1, 2, 3]), 'f': np.array([1, 2, 3]), 'g': jnp.array(1.0), 'h': np.array(1.0)})
        logger.log()
        print(logger.raw_logs())
        logger.accum_logs({'a': 17, 'b': 6, 'c': [5, 6, 9], 'd': 3.0, 'e': jnp.array([4, 5, 6]), 'f': np.array([4, 5, 6]), 'g': jnp.array(2.0), 'h': np.array(2.0)})
        logger.log()
        print(logger.raw_logs())
        logger.reset_logs()
        logger.accum_logs({'a': 1})
        logger.log()
    # print(jax.tree_util.build_tree(jax.tree, 0))
    # print(jax.tree_util.tree_flatten({'a': 1, 'b': 2})[1].children()[0])
