#!/usr/bin/env python3
"""
Adaptation of the MTSAC implementation based on Metaworld. Benchmarked on MTFlexible.
Original paper: https://arxiv.org/pdf/1910.10897.pdf
Compared to the paper we do not support TE-PPO as it relies on Tensorflow.
We provide a Pytorch only implementation.
"""

# Disable warnings (use at your own risk)
import warnings
warnings.filterwarnings("ignore")

import click
import numpy as np
import psutil
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import deterministic
from garage.experiment.task_sampler import MetaWorldTaskSampler
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler, RaySampler
from garage.torch import set_gpu_mode
from garage.torch.algos import MTSAC, PPO, TRPO
from garage.torch.policies import TanhGaussianMLPPolicy, GaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

from metaworld_additional_envs import MTFlexible

AWS_SHUTDOWN_AVAILABLE = False
try:
    from toolbox.aws import shutdown
    AWS_SHUTDOWN_AVAILABLE = True
except ImportError:
    print("AWS shutdown not available.")


@click.command()
# Reproducibility
@click.option('--seed', 'seed', type=int, default=1)
@click.option('--shutdown', 'shutdown', type=bool, default=False, help="Shutdown the AWS instance after the experiment.")

# Environment
@click.option('--n_tasks', 'n_tasks', type=int, default=3, help="Number of tasks to use in the benchmarks. Use 10 for MT10 and 3 for MT3.")
@click.option('--increasing_difficulty', 'increasing_difficulty', type=bool, default=True, help="Whether to use the tasks in increasing difficulty order or not.")
@click.option('--task_name', 'task_name', type=str, default="", help="Task name if n_tasks == 1.")

# Parallelism
@click.option('--sampler', 'sampler', type=str, default="local", help="Sampler to use. Can be either local or ray.")
@click.option('--n_workers', type=int, default=-1, help="Number of workers to use. If -1, it will be set to psutil.cpu_count(logical=False)")
@click.option('--n_envs', 'n_envs', type=int, default=2, help="Number of environments per worker. Each environment is approximately ~50mb large. So n_tasks * n_parallel * n_envs * 50mb should give you an idea of the memory usage.")

# RL parameters
@click.option('--algo', 'algo', type=str, default="mtsac", help="Algorithm to use. Can be either mtsac, ppo or trpo.")
@click.option('--discount', 'discount', type=float, default=0.99)

# Policy and Q function parameters
@click.option('--n_hidden_layers', 'n_hidden_layers', type=int, default=3)
@click.option('--size_hidden_layers', 'size_hidden_layers', type=int, default=400)
@click.option('--replay_buffer_size', 'replay_buffer_size', type=int, default=int(1e6))

# Training parameters
@click.option('--timesteps', 'timesteps', type=int, default=20000000)
@click.option('--num_training_batch_before_eval', 'num_training_batch_before_eval', type=int, default=8)
@click.option('--use_gpu', 'use_gpu', type=bool, default=True)
@click.option('--batch_size', 'batch_size', type=int, default=-1, help="Batch size. If -1, it will be set to int(env.spec.max_episode_length * n_workers).")

# Utils
@click.option('--render_env', 'render_env', type=bool, default=False, help="Render the environment during training.")
@click.option('--use_wandb', 'use_wandb', type=bool, default=False, help="Log with wandb.")
@click.option('--run_name', 'run_name', type=str, default="", help="Name of the run. Only used when log_with is wandb.")
@click.option('--run_notes', 'run_notes', type=str, default="", help="Notes of the run. Only used when log_with is wandb.")

# Improvements
@click.option('--use_pcgrad', 'use_pcgrad', type=bool, default=False, help="Whether to use PCGrad or not.")
@click.option('--use_proj_layer', 'use_proj_layer', type=bool, default=False, help="Whether to use projected layers or not.")
@click.option('--proj_layer_size', 'proj_layer_size', type=int, default=0, help="Size of the projected layers.")
@click.option('--sampling_alpha', 'sampling_alpha', type=float, default=0.5, help="Alpha parameter for the sampling distribution.")
@click.option('--sampling_beta', 'sampling_beta', type=float, default=0.5, help="Beta parameter for the sampling distribution.")
@click.option('--sampling_gamma', 'sampling_gamma', type=float, default=1.0, help="Gamma parameter for the sampling distribution.")
@click.option('--sampling_delta', 'sampling_delta', type=float, default=0.8, help="Delta parameter for the sampling distribution.")
@click.option('--sampling_epsilon', 'sampling_epsilon', type=float, default=0.2, help="Epsilon parameter for the sampling distribution.")
@wrap_experiment(snapshot_mode='gap', snapshot_gap=50)
def metaworld_mtf(ctxt=None, *,
                    seed: int,
                    shutdown: bool,
                    n_tasks: int,
                    increasing_difficulty: bool,
                    task_name: str,
                    sampler: str,
                    n_workers: int,
                    n_envs: int,
                    algo: str,
                    discount: float,
                    n_hidden_layers: int,
                    size_hidden_layers: int,
                    replay_buffer_size: int,
                    timesteps: int,
                    num_training_batch_before_eval: int,
                    use_gpu: bool,
                    batch_size: int,
                    render_env: bool,
                    use_wandb: bool,
                    run_name: str,
                    run_notes: str,
                    use_pcgrad: bool,
                    use_proj_layer: bool,
                    proj_layer_size: int,
                    sampling_alpha: float,
                    sampling_beta: float,
                    sampling_gamma: float,
                    sampling_delta: float,
                    sampling_epsilon: float):
    """Train either MTSAC, PPO or TRPO with MTFlexible environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        all other arguments are passed to the click decorators.
    """

    # Verify arguments
    if task_name == "":
        task_name = None
    assert n_tasks <= 500, "n_tasks must be <= 500"
    assert num_training_batch_before_eval > 0, "num_training_batch_before_eval must be > 0"
    assert algo in ["mtsac", "ppo", "trpo"], "algo must be either mtsac, ppo or trpo"
    assert sampler in ["local", "ray"], "sampler must be either local or ray"
    if n_tasks > 1:
        assert task_name is None, "task_name must be None if n_tasks > 1"
    else:
        assert task_name is not None, "task_name must be not None if n_tasks == 1"

    # Set up experiment
    deterministic.set_seed(seed)
    trainer = Trainer(ctxt)
    mtf = MTFlexible(n=n_tasks, task_name=task_name)  # pylint: disable=no-member
    mtf_test = MTFlexible(n=n_tasks, task_name=task_name)  # pylint: disable=no-member

    # pylint: disable=missing-return-doc, missing-return-type-doc
    def wrap(env, _):
        return normalize(env, normalize_reward=True)

    train_task_sampler = MetaWorldTaskSampler(mtf,
                                              'train',
                                              wrap,
                                              add_env_onehot=True)
    test_task_sampler = MetaWorldTaskSampler(mtf_test,
                                             'train',
                                             add_env_onehot=True)
    mtf_train_envs = train_task_sampler.sample(n_tasks)
    env = mtf_train_envs[0]()
    mtf_test_envs = [env_up() for env_up in test_task_sampler.sample(n_tasks)]
    

    # Set default args
    if n_workers < 0:
        n_workers = psutil.cpu_count(logical=False)
        if algo == "mtsac":
            n_workers = n_tasks
    if batch_size < 0:
        batch_size = int(env.spec.max_episode_length * n_workers)
    epochs = timesteps // batch_size

    # Print arguments
    print("============= Arguments =============")
    print("seed: {}".format(seed))
    print("shutdown: {}".format(shutdown))
    print('')
    print("n_tasks: {}".format(n_tasks))
    print("increasing_difficulty: {}".format(increasing_difficulty))
    print("task_name: {}".format(task_name))
    print('')
    print("sampler: {}".format(sampler))
    print("n_workers: {}".format(n_workers))
    print("n_envs: {}".format(n_envs))
    print('')
    print("algo: {}".format(algo))
    print("discount: {}".format(discount))
    print('')
    print("n_hidden_layers: {}".format(n_hidden_layers))
    print("size_hidden_layers: {}".format(size_hidden_layers))
    print("replay_buffer_size: {}".format(replay_buffer_size))
    print('')
    print("timesteps: {}".format(timesteps))
    print("num_training_batch_before_eval: {} (MTSAC only)".format(num_training_batch_before_eval))
    print("use_gpu: {}".format(use_gpu))
    print("batch_size: {}".format(batch_size))
    print("epochs: {}".format(epochs))
    print('')
    print("render_env: {}".format(render_env))
    print("use_wandb: {}".format(use_wandb))
    print("run_name: {}".format(run_name))
    print("run_notes: {}".format(run_notes))
    print('')
    print("use_pcgrad: {}".format(use_pcgrad))
    print("use_proj_layer: {}".format(use_proj_layer))
    print("proj_layer_size: {}".format(proj_layer_size))
    print("sampling_alpha: {}".format(sampling_alpha))
    print("sampling_beta: {}".format(sampling_beta))
    print("sampling_gamma: {}".format(sampling_gamma))
    print("sampling_delta: {}".format(sampling_delta))
    print("sampling_epsilon: {}".format(sampling_epsilon))
    print("======================================")

    # Log with wandb
    if use_wandb:
        import wandb
        from toolbox.log import tabular
        wandb.init(
            project="metaworld",
            name=run_name if run_name != "" else None,
            notes=run_notes,
            config={
                "seed": seed,
                "n_tasks": n_tasks,
                "increasing_difficulty": increasing_difficulty,
                "task_name": task_name,
                "sampler": sampler,
                "n_workers": n_workers,
                "n_envs": n_envs,
                "algo": algo,
                "discount": discount,
                "n_hidden_layers": n_hidden_layers,
                "size_hidden_layers": size_hidden_layers,
                "replay_buffer_size": replay_buffer_size,
                "timesteps": timesteps,
                "num_training_batch_before_eval": num_training_batch_before_eval,
                "use_gpu": use_gpu,
                "batch_size": batch_size,
                "epochs": epochs,
                "render_env": render_env,
                "use_pcgrad": use_pcgrad,
                "use_proj_layer": use_proj_layer,
                "proj_layer_size": proj_layer_size,
                "sampling_alpha": sampling_alpha,
                "sampling_beta": sampling_beta,
                "sampling_gamma": sampling_gamma,
                "sampling_delta": sampling_delta,
                "sampling_epsilon": sampling_epsilon,
            }
        )
        tabular.set_wandb(use_wandb=True, wandb_step_factor=batch_size*num_training_batch_before_eval)

    policy, qf1, qf2, value_function = None, None, None, None
    hidden_sizes = [size_hidden_layers] * n_hidden_layers
    if algo == "mtsac":
        policy = TanhGaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=nn.ReLU,
            output_nonlinearity=None,
            min_std=np.exp(-20.),
            max_std=np.exp(2.),
        )

        qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=hidden_sizes,
                                    hidden_nonlinearity=F.relu)

        qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=hidden_sizes,
                                    hidden_nonlinearity=F.relu)
    elif algo == "ppo" or algo == "trpo":
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=torch.tanh,
            output_nonlinearity=None,
        )

        value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                                hidden_sizes=hidden_sizes,
                                                hidden_nonlinearity=torch.tanh,
                                                output_nonlinearity=None)


    replay_buffer = PathBuffer(capacity_in_transitions=replay_buffer_size, )

    sampler_object = None
    if sampler == "local":
        print("Using local sampler")
        sampler_object = LocalSampler(
            agents=policy,
            envs=mtf_train_envs,
            max_episode_length=env.spec.max_episode_length,
            n_workers=n_tasks,
            worker_class=FragmentWorker,
            worker_args=dict(n_envs=n_envs))
    elif sampler == "ray":
        print("Using ray sampler")
        sampler_object = RaySampler(agents=policy,
                            envs=env,
                            max_episode_length=env.spec.max_episode_length,
                            n_workers=n_workers)
    else:
        raise ValueError("sampler must be either local or ray")
    

    algo_object = None
    if algo == "mtsac":
        epochs = epochs // num_training_batch_before_eval
        print(f"Using MTSAC. Evaluation every {num_training_batch_before_eval} training batches of size {batch_size}. Total of {epochs} epochs.")
        assert policy is not None
        assert qf1 is not None
        assert qf2 is not None

        algo_object = MTSAC(policy=policy,
                    qf1=qf1,
                    qf2=qf2,
                    sampler=sampler_object,
                    gradient_steps_per_itr=env.spec.max_episode_length,
                    eval_env=mtf_test_envs,
                    env_spec=env.spec,
                    num_tasks=n_tasks,
                    steps_per_epoch=num_training_batch_before_eval,
                    replay_buffer=replay_buffer,
                    min_buffer_size=1500,
                    target_update_tau=5e-3,
                    discount=discount,
                    buffer_batch_size=1280,
                    # The tasks we are using are not randomized, so if the agent succeeds once, it will succeed always.
                    # (as long as the agent weights are not updated)
                    # This means that there is not need to evaluate the agent multiple times per epoch.
                    num_evaluation_episodes=1,
                    render_env=render_env,
                    sampling_alpha=sampling_alpha,
                    sampling_beta=sampling_beta,
                    sampling_gamma=sampling_gamma,
                    sampling_delta=sampling_delta,
                    sampling_epsilon=sampling_epsilon)
    elif algo == "ppo":
        print("Using PPO")
        assert policy is not None
        assert value_function is not None

        algo_object = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler_object,
               discount=discount,
               gae_lambda=0.95,
               center_adv=True,
               lr_clip_range=0.2)
    elif algo == "trpo":
        print("Using TRPO")
        assert policy is not None
        assert value_function is not None

        algo_object = TRPO(env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                sampler=sampler_object,
                discount=discount,
                gae_lambda=0.95)
    else:
        raise ValueError("algo must be either mtsac, ppo or trpo")

    if use_gpu:      
        set_gpu_mode(True)
    algo_object.to()
    trainer.setup(algo=algo_object, env=mtf_train_envs)
    trainer.train(n_epochs=epochs, batch_size=batch_size)

    if use_wandb:
        wandb.finish()

    if shutdown and AWS_SHUTDOWN_AVAILABLE:
        shutdown()


# pylint: disable=missing-kwoa
metaworld_mtf()
