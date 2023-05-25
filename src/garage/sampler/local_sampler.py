"""Sampler that runs workers in the main process."""
import copy

import numpy as np
import psutil
from typing import List, Optional

from garage import EpisodeBatch
from garage.experiment.deterministic import get_seed
from garage.sampler.default_worker import DefaultWorker
from garage.sampler.sampler import Sampler
from garage.sampler.worker_factory import WorkerFactory


class LocalSampler(Sampler):
    """Sampler that runs workers in the main process.

    This is probably the simplest possible sampler. It's called the "Local"
    sampler because it runs everything in the same process and thread as where
    it was called from.

    The sampler need to be created either from a worker factory or from
    parameters which can construct a worker factory. See the __init__ method
    of WorkerFactory for the detail of these parameters.

    Args:
        agents (Policy or List[Policy]): Agent(s) to use to sample episodes.
            If a list is passed in, it must have length exactly
            `worker_factory.n_workers`, and will be spread across the
            workers.
        envs (Environment or List[Environment]): Environment from which
            episodes are sampled. If a list is passed in, it must have length
            exactly `worker_factory.n_workers`, and will be spread across the
            workers.
        worker_factory (WorkerFactory): Pickleable factory for creating
            workers. Should be transmitted to other processes / nodes where
            work needs to be done, then workers should be constructed
            there. Either this param or params after this are required to
            construct a sampler.
        max_episode_length(int): Params used to construct a worker factory.
            The maximum length episodes which will be sampled.
        is_tf_worker (bool): Whether it is workers for TFTrainer.
        seed(int): The seed to use to initialize random number generators.
        n_workers(int): The number of workers to use.
        worker_class(type): Class of the workers. Instances should implement
            the Worker interface.
        worker_args (dict or None): Additional arguments that should be passed
            to the worker.

    """

    def __init__(
            self,
            agents,
            envs,
            *,  # After this require passing by keyword.
            worker_factory=None,
            max_episode_length=None,
            is_tf_worker=False,
            seed=get_seed(),
            n_workers=psutil.cpu_count(logical=False),
            worker_class=DefaultWorker,
            worker_args=None):
        # pylint: disable=super-init-not-called
        if worker_factory is None and max_episode_length is None:
            raise TypeError('Must construct a sampler from WorkerFactory or'
                            'parameters (at least max_episode_length)')
        if isinstance(worker_factory, WorkerFactory):
            self._factory = worker_factory
        else:
            self._factory = WorkerFactory(
                max_episode_length=max_episode_length,
                is_tf_worker=is_tf_worker,
                seed=seed,
                n_workers=n_workers,
                worker_class=worker_class,
                worker_args=worker_args)

        self._agents = self._factory.prepare_worker_messages(agents)
        self._envs = self._factory.prepare_worker_messages(
            envs, preprocess=copy.deepcopy)
        self._workers = [
            self._factory(i) for i in range(self._factory.n_workers)
        ]
        for worker, agent, env in zip(self._workers, self._agents, self._envs):
            worker.update_agent(agent)
            worker.update_env(env)
        self.total_env_steps = 0

    @classmethod
    def from_worker_factory(cls, worker_factory, agents, envs):
        """Construct this sampler.

        Args:
            worker_factory (WorkerFactory): Pickleable factory for creating
                workers. Should be transmitted to other processes / nodes where
                work needs to be done, then workers should be constructed
                there.
            agents (Agent or List[Agent]): Agent(s) to use to sample episodes.
                If a list is passed in, it must have length exactly
                `worker_factory.n_workers`, and will be spread across the
                workers.
            envs (Environment or List[Environment]): Environment from which
                episodes are sampled. If a list is passed in, it must have
                length exactly `worker_factory.n_workers`, and will be spread
                across the workers.

        Returns:
            Sampler: An instance of `cls`.

        """
        return cls(agents, envs, worker_factory=worker_factory)

    def _update_workers(self, agent_update, env_update):
        """Apply updates to the workers.

        Args:
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        """
        agent_updates = self._factory.prepare_worker_messages(agent_update)
        env_updates = self._factory.prepare_worker_messages(
            env_update, preprocess=copy.deepcopy)
        for worker, agent_up, env_up in zip(self._workers, agent_updates,
                                            env_updates):
            worker.update_agent(agent_up)
            worker.update_env(env_up)

    def obtain_samples(self, itr, num_samples, agent_update, env_update=None, task_distribution: Optional[np.ndarray] = None, return_sample_distribution: bool = False):
        """Collect at least a given number transitions (timesteps).

        Args:
            itr(int): The current iteration number. Using this argument is
                deprecated.
            num_samples (int): Minimum number of transitions / timesteps to
                sample.
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            task_distribution (Optional[np.ndarray]): Only relevant for multitask
                environments. If provided, the tasks will be distributed
                according to this distribution. If not provided, the tasks
                will be distributed uniformly.
            return_sample_distribution (bool): Whether to return the sample
                distribution (i.e. the number of samples per worker)

        Returns:
            EpisodeBatch: The batch of collected episodes.

        """
        self._update_workers(agent_update, env_update)
        batches = []
        completed_samples = 0

        samples_per_worker = [0] * len(self._workers)
        continue_sampling = True
        while continue_sampling:
            # If the task distribution is uniform (checked by computing the standard deviation)
            # we can round robin the workers to get a more uniform distribution
            # which is what is done in the original implementation (the else below)
            if task_distribution is not None and np.std(task_distribution) > 1e-6:
                # Here we assume that there is one worker per task
                # We sample worker in self._workers with probability
                # task_distribution[i] (the sum is already normalized)
                idx: int = np.random.choice(len(self._workers), p=task_distribution)
                worker = self._workers[idx]
                batch = worker.rollout()
                completed_samples += len(batch.actions)
                samples_per_worker[idx] += len(batch.actions)
                batches.append(batch)
                if completed_samples >= num_samples:
                    continue_sampling = False
                    break
            else:
                for idx, worker in enumerate(self._workers):
                    batch = worker.rollout()
                    completed_samples += len(batch.actions)
                    samples_per_worker[idx] += len(batch.actions)
                    batches.append(batch)
                    if completed_samples >= num_samples:
                        continue_sampling = False
                        break
                    
        assert completed_samples >= num_samples, "Not enough samples collected"
        samples = EpisodeBatch.concatenate(*batches)
        self.total_env_steps += sum(samples.lengths)
        if return_sample_distribution:
            return samples, samples_per_worker
        return samples

    def obtain_exact_episodes(self,
                              n_eps_per_worker,
                              agent_update,
                              env_update=None):
        """Sample an exact number of episodes per worker.

        Args:
            n_eps_per_worker (int): Exact number of episodes to gather for
                each worker.
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before samplin episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        Returns:
            EpisodeBatch: Batch of gathered episodes. Always in worker
                order. In other words, first all episodes from worker 0,
                then all episodes from worker 1, etc.

        """
        self._update_workers(agent_update, env_update)
        batches = []
        for worker in self._workers:
            for _ in range(n_eps_per_worker):
                batch = worker.rollout()
                batches.append(batch)
        samples = EpisodeBatch.concatenate(*batches)
        self.total_env_steps += sum(samples.lengths)
        return samples

    def shutdown_worker(self):
        """Shutdown the workers."""
        for worker in self._workers:
            worker.shutdown()

    def __getstate__(self):
        """Get the pickle state.

        Returns:
            dict: The pickled state.

        """
        state = self.__dict__.copy()
        # Workers aren't picklable (but WorkerFactory is).
        state['_workers'] = None
        return state

    def __setstate__(self, state):
        """Unpickle the state.

        Args:
            state (dict): Unpickled state.

        """
        self.__dict__.update(state)
        self._workers = [
            self._factory(i) for i in range(self._factory.n_workers)
        ]
        for worker, agent, env in zip(self._workers, self._agents, self._envs):
            worker.update_agent(agent)
            worker.update_env(env)
