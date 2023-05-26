"""This modules creates a MTSAC model in PyTorch."""
# yapf: disable
import numpy as np
import torch
from tqdm import tqdm
from typing import Dict
from toolbox.log import tabular

from garage import (EpisodeBatch, log_multitask_performance,
                    obtain_evaluation_episodes, StepType)
from garage.torch import global_device
from garage.torch.algos import SAC
from sympy.stats.sampling.tests.test_sample_continuous_rv import test_sampling_gamma_inverse
from matplotlib.colors import same_color

# yapf: enable


class MTSAC(SAC):
    """A MTSAC Model in Torch.

    This MTSAC implementation uses is the same as SAC except for a small change
    called "disentangled alphas". Alpha is the entropy coefficient that is used
    to control exploration of the agent/policy. Disentangling alphas refers to
    having a separate alpha coefficients for every task learned by the policy.
    The alphas are accessed by using a the one-hot encoding of an id that is
    assigned to each task.

    Args:
        policy (garage.torch.policy.Policy): Policy/Actor/Agent that is being
            optimized by SAC.
        qf1 (garage.torch.q_function.ContinuousMLPQFunction): QFunction/Critic
            used for actor/policy optimization. See Soft Actor-Critic and
            Applications.
        qf2 (garage.torch.q_function.ContinuousMLPQFunction): QFunction/Critic
            used for actor/policy optimization. See Soft Actor-Critic and
            Applications.
        replay_buffer (ReplayBuffer): Stores transitions that are previously
            collected by the sampler.
        env_spec (EnvSpec): The env_spec attribute of the environment that the
            agent is being trained in.
        sampler (garage.sampler.Sampler): Sampler.
        num_tasks (int): The number of tasks being learned.
        max_episode_length_eval (int or None): Maximum length of episodes used
            for off-policy evaluation. If None, defaults to
            `max_episode_length`.
        eval_env (Environment): The environment used for collecting evaluation
            episodes.
        gradient_steps_per_itr (int): Number of optimization steps that should
            occur before the training step is over and a new batch of
            transitions is collected by the sampler.
        fixed_alpha (float): The entropy/temperature to be used if temperature
            is not supposed to be learned.
        target_entropy (float): target entropy to be used during
            entropy/temperature optimization. If None, the default heuristic
            from Soft Actor-Critic Algorithms and Applications is used.
        initial_log_entropy (float): initial entropy/temperature coefficient
            to be used if a fixed_alpha is not being used (fixed_alpha=None),
            and the entropy/temperature coefficient is being learned.
        discount (float): The discount factor to be used during sampling and
            critic/q_function optimization.
        buffer_batch_size (int): The number of transitions sampled from the
            replay buffer that are used during a single optimization step.
        min_buffer_size (int): The minimum number of transitions that need to
            be in the replay buffer before training can begin.
        target_update_tau (float): A coefficient that controls the rate at
            which the target q_functions update over optimization iterations.
        policy_lr (float): Learning rate for policy optimizers.
        qf_lr (float): Learning rate for q_function optimizers.
        reward_scale (float): Reward multiplier. Changing this hyperparameter
            changes the effect that the reward from a transition will have
            during optimization.
        optimizer (torch.optim.Optimizer): Optimizer to be used for
            policy/actor, q_functions/critics, and temperature/entropy
            optimizations.
        steps_per_epoch (int): Number of train_once calls per epoch.
        num_evaluation_episodes (int): The number of evaluation episodes used
            for computing eval stats at the end of every epoch.
        use_deterministic_evaluation (bool): True if the trained policy
            should be evaluated deterministically.

    """

    def __init__(
        self,
        policy,
        qf1,
        qf2,
        replay_buffer,
        env_spec,
        sampler,
        *,
        num_tasks,
        eval_env,
        gradient_steps_per_itr,
        max_episode_length_eval=None,
        fixed_alpha=None,
        target_entropy=None,
        initial_log_entropy=0.,
        discount=0.99,
        buffer_batch_size=64,
        min_buffer_size=int(1e4),
        target_update_tau=5e-3,
        policy_lr=3e-4,
        qf_lr=3e-4,
        reward_scale=1.0,
        optimizer=torch.optim.Adam,
        steps_per_epoch=1,
        num_evaluation_episodes=5,
        use_deterministic_evaluation=True,
        render_env = False,
        # α: Factor of the success rate coefficient that is used to sample
        # each task. The coefficient is proportional to
        # (1 + ε - capped_filtered_success_rate[task]) ** γ
        # with capped_filtered_success_rate[task] = 0 if filtered_success_rate[task] < ε
        # and capped_filtered_success_rate[task] = filtered_success_rate[task] otherwise
        sampling_alpha = 0.5,
        # β: Factor of the undiscounted return coefficient that is used to sample
        # each task. The coefficient is proportional to
        # (1000/(filtered_avg_discounted_return[task] + 50)) ** γ
        sampling_beta = 0.5,
        # γ: Exponent used for the coefficients
        sampling_gamma = 1.0,
        # δ: Filter coefficient for the success rate and the average discounted return
        # filtered_success_rate[task] = δ * filtered_success_rate[task] + (1 - δ) * success_rate[task]
        # Same for filtered_avg_discounted_return
        sampling_delta = 0.8,
        # ε: Minimum probability of sampling each task: ε / num_tasks
        # Also used to compute capped_filtered_success_rate
        sampling_epsilon = 0.2,
    ):

        super().__init__(
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            replay_buffer=replay_buffer,
            sampler=sampler,
            env_spec=env_spec,
            max_episode_length_eval=max_episode_length_eval,
            gradient_steps_per_itr=gradient_steps_per_itr,
            fixed_alpha=fixed_alpha,
            target_entropy=target_entropy,
            initial_log_entropy=initial_log_entropy,
            discount=discount,
            buffer_batch_size=buffer_batch_size,
            min_buffer_size=min_buffer_size,
            target_update_tau=target_update_tau,
            policy_lr=policy_lr,
            qf_lr=qf_lr,
            reward_scale=reward_scale,
            optimizer=optimizer,
            steps_per_epoch=steps_per_epoch,
            num_evaluation_episodes=num_evaluation_episodes,
            eval_env=eval_env,
            use_deterministic_evaluation=use_deterministic_evaluation)
        # Task sampling based on success rate parameters
        self._sampling_alpha = sampling_alpha
        self._sampling_beta = sampling_beta
        self._sampling_gamma = sampling_gamma
        self._sampling_filter_factor = sampling_delta
        self._sampling_epsilon = sampling_epsilon

        self._num_tasks = num_tasks
        self._eval_env = eval_env
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha
        if self._use_automatic_entropy_tuning:
            if target_entropy:
                self._target_entropy = target_entropy
            else:
                self._target_entropy = -np.prod(
                    self.env_spec.action_space.shape).item()
            self._log_alpha = torch.Tensor([self._initial_log_entropy] *
                                           self._num_tasks).requires_grad_()
            self._alpha_optimizer = optimizer([self._log_alpha] *
                                              self._num_tasks,
                                              lr=self._policy_lr)
        else:
            self._log_alpha = torch.Tensor([self._fixed_alpha] *
                                           self._num_tasks).log()
        self._epoch_mean_success_rate = []
        self._epoch_median_success_rate = []
        self._render_env = render_env

    def _get_log_alpha(self, samples_data):
        """Return the value of log_alpha.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Raises:
            ValueError: If the number of tasks, num_tasks passed to
                this algorithm doesn't match the length of the task
                one-hot id in the observation vector.

        Returns:
            torch.Tensor: log_alpha. shape is (1, self.buffer_batch_size)

        """
        obs = samples_data['observation']
        log_alpha = self._log_alpha
        one_hots = obs[:, -self._num_tasks:]
        if (log_alpha.shape[0] != one_hots.shape[1]
                or one_hots.shape[1] != self._num_tasks
                or log_alpha.shape[0] != self._num_tasks):
            raise ValueError(
                'The number of tasks in the environment does '
                'not match self._num_tasks. Are you sure that you passed '
                'The correct number of tasks?')
        ret = torch.mm(one_hots, log_alpha.unsqueeze(0).t()).squeeze()
        return ret

    def _evaluate_policy(self, epoch):
        """Evaluate the performance of the policy via deterministic sampling.

            Statistics such as (average) discounted return and success rate are
            recorded.

        Args:
            epoch (int): The current training epoch.

        Returns:
            Dict[str, Dict[str, float]]: A dict containing statistics
                about evaluation (see log_multitask_performance)

        """
        eval_eps = []
        for eval_env in self._eval_env:
            eval_eps.append(
                obtain_evaluation_episodes(
                    self.policy,
                    eval_env,
                    self._max_episode_length_eval,
                    num_eps=self._num_evaluation_episodes,
                    render_env = self._render_env,
                    deterministic=self._use_deterministic_evaluation))
        eval_eps = EpisodeBatch.concatenate(*eval_eps)
        performance = log_multitask_performance(epoch, eval_eps,
                                                self._discount)
        return performance

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        super().to(device)
        if device is None:
            device = global_device()
        if not self._use_automatic_entropy_tuning:
            self._log_alpha = torch.Tensor([self._fixed_alpha] *
                                           self._num_tasks).log().to(device)
        else:
            self._log_alpha = torch.Tensor(
                [self._initial_log_entropy] *
                self._num_tasks).to(device).requires_grad_()
            self._alpha_optimizer = self._optimizer([self._log_alpha],
                                                    lr=self._policy_lr)
    

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        if not self._eval_env:
            self._eval_env = trainer.get_env_copy()
        last_return = None
        epoch_count = 0
        filtered_success_rate = [0.0] * self._num_tasks
        filtered_avg_discounted_return = [0.0] * self._num_tasks
        task_names = []
        for _ in trainer.step_epochs():
            epoch_count += 1
            for _ in tqdm(range(self._steps_per_epoch), desc=f"Epoch {epoch_count}"):
                if not (self.replay_buffer.n_transitions_stored >=
                        self._min_buffer_size):
                    batch_size = int(self._min_buffer_size)
                else:
                    batch_size = None
                # Sample each task proportionally to (1 - success_rate) ** sampling_factor_success_rate
                # coefficient = np.array([1. + ε - filtered_success_rate[i] for i in range(self._num_tasks)]) ** self._sampling_factor_success_rate
                # Sample each task proportionally to (1000/(avg_discounted_return + 50)) ** sampling_factor_success_rate
                # Then the heuristic are normalized, making sure that the sum of the coefficient is 1.
                # We then add a second term: coefficient_success that is proportional to
                # (1. + ε - x[i]) ** sampling_factor_success_rate
                # with x[i] = 0. if filtered_success_rate[i] < ε, and x[i] = filtered_success_rate[i] otherwise
                # The final coefficient is the sum of the two terms, normalized to sum to 1.
                # Additionally, each task must probability of at least ε / self._num_tasks
                coefficient_return = np.array([1000.0 / (filtered_avg_discounted_return[i] + 50.0) for i in range(self._num_tasks)]) ** self._sampling_gamma
                coefficient_return = coefficient_return / np.sum(coefficient_return)
                x = np.array([0.0 if filtered_success_rate[i] < self._sampling_epsilon else filtered_success_rate[i] for i in range(self._num_tasks)])
                coefficient_success = np.array([1. + self._sampling_epsilon - x[i] for i in range(self._num_tasks)]) ** self._sampling_gamma
                coefficient_success = coefficient_success / np.sum(coefficient_success)
                coefficient = (self._sampling_beta * coefficient_return + self._sampling_alpha * coefficient_success) / (self._sampling_alpha + self._sampling_beta)
                coefficient = (1 - self._sampling_epsilon) * coefficient / np.sum(coefficient)
                coefficient += self._sampling_epsilon / self._num_tasks
                assert np.abs(np.sum(coefficient) - 1.0) < 1e-6

                trainer.step_episode, sample_distribution = trainer.obtain_samples(
                    trainer.step_itr, batch_size, task_distribution = coefficient, return_sample_distribution=True)
                # Log the coefficient for each task if we have the task names
                if len(task_names) == self._num_tasks:
                    for idx, task in enumerate(task_names):
                        with tabular.prefix(task + '/'):
                            tabular.record('SamplingProbabilityCombined', coefficient[idx], step=epoch_count-1)
                            tabular.record('SamplingProbabilityReturn', coefficient_return[idx], step=epoch_count-1)
                            tabular.record('SamplingProbabilitySuccess', coefficient_success[idx], step=epoch_count-1)
                            tabular.record('SampleDistribution', sample_distribution[idx], step=epoch_count-1)
                path_returns = []
                for path in trainer.step_episode:
                    self.replay_buffer.add_path(
                        dict(observation=path['observations'],
                             action=path['actions'],
                             reward=path['rewards'].reshape(-1, 1),
                             next_observation=path['next_observations'],
                             terminal=np.array([
                                 step_type == StepType.TERMINAL
                                 for step_type in path['step_types']
                             ]).reshape(-1, 1)))
                    path_returns.append(sum(path['rewards']))
                assert len(path_returns) == len(trainer.step_episode)
                self.episode_rewards.append(np.mean(path_returns))
                for _ in range(self._gradient_steps):
                    policy_loss, qf1_loss, qf2_loss = self.train_once()
            performance: Dict[str, Dict[str, float]] = self._evaluate_policy(trainer.step_itr)
            last_return = performance['average']['average_return']
            # For each task, we get the success_rate (exclude average from performance)
            if len(task_names) == 0:
                for task in performance.keys():
                    if task != 'average':
                        task_names.append(task)
            success_rate = [performance[task]['success_rate'] for task in task_names]
            filtered_success_rate = [self._sampling_filter_factor * filtered_success_rate[i] + (1 - self._sampling_filter_factor) * success_rate[i] for i in range(self._num_tasks)]
            avg_discounted_return = [performance[task]['average_discounted_return'] for task in task_names]
            filtered_avg_discounted_return = [self._sampling_filter_factor * filtered_avg_discounted_return[i] + (1 - self._sampling_filter_factor) * avg_discounted_return[i] for i in range(self._num_tasks)]

            for idx, task in enumerate(task_names):
                with tabular.prefix(task + '/'):
                    tabular.record('FilteredSuccessRate', filtered_success_rate[idx], step=epoch_count-1)
                    tabular.record('FilteredAvgDiscountedReturn', filtered_avg_discounted_return[idx], step=epoch_count-1)
            self._log_statistics(policy_loss, qf1_loss, qf2_loss)
            tabular.record('TotalEnvSteps', trainer.total_env_steps, step=epoch_count-1)
            trainer.step_itr += 1

        return last_return
