import time
import sys
import csv

import gym
import numpy as np
import tensorflow as tf

from stable_baselines import logger
from stable_baselines.common import explained_variance, tf_util, ActorCriticRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.policies import ActorCriticPolicy, RecurrentActorCriticPolicy
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.schedules import Scheduler
from stable_baselines.common.tf_util import mse, total_episode_reward_logger, calc_entropy
from stable_baselines.common.math_util import safe_mean

from collections import deque

def discount_with_dones(rewards, dones, gamma):    ## Same function with LIRPG/baselines/a2c/utils.py
    """
    Apply the discount value to the reward, where the environment is not done

    :param rewards: ([float]) The rewards
    :param dones: ([bool]) Whether an environment is done or not
    :param gamma: (float) The discount value
    :return: ([float]) The discounted rewards
    """
    discounted = []
    ret = 0  # Return: discounted reward
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + gamma * ret * (1. - done)  # fixed off by one bug
        discounted.append(ret)
    return discounted[::-1]


class A2C(ActorCriticRLModel):
    """
    The A2C (Advantage Actor Critic) model class, https://arxiv.org/abs/1602.01783

    :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) Discount factor
    :param n_steps: (int) The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param vf_coef: (float) Value function coefficient for the loss calculation
    :param ent_coef: (float) Entropy coefficient for the loss calculation
    :param max_grad_norm: (float) The maximum value for the gradient clipping
    :param learning_rate: (float) The learning rate

    :param alpha: (float)  RMSProp decay parameter (default: 0.99)
    :param momentum: (float) RMSProp momentum parameter (default: 0.0)
    :param epsilon: (float) RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update)
        (default: 1e-5)

    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                              'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
                              (used only for loading)
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        WARNING: this logging can take a lot of space quickly
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """
    
    def __init__(self, policy, env, gamma=0.99, n_steps=5, #vf_coef=0.25,
                 v_mix_coef=0.5, v_ex_coef=1.0, r_ex_coef=1, r_in_coef=0.005,
                 ent_coef=0.01, max_grad_norm=0.5,
                 #learning_rate=7e-4,
                 lr_alpha = 7e-4, lr_beta = 7e-4,
                 alpha=0.99, momentum=0.0, epsilon=1e-5, lr_schedule='constant',
                 verbose=0, tensorboard_log=None, #'C:/Users/Zihang Guan/Desktop/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020-master/results/tb',
                 _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False, seed=None, n_cpu_tf_sess=None):
#use lr_alpha and lr_beta to replace learning_rate
        self.n_steps = n_steps
        self.gamma = gamma
        #self.vf_coef = vf_coef
        self.v_mix_coef = v_mix_coef
        self.v_ex_coef = v_ex_coef
        self.r_ex_coef = r_ex_coef
        self.r_in_coef = r_in_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.alpha = alpha
        self.momentum = momentum
        self.epsilon = epsilon
        self.lr_schedule = lr_schedule
        #self.learning_rate = learning_rate
        self.lr_alpha = lr_alpha
        self.lr_beta = lr_beta
        self.tensorboard_log = tensorboard_log
        self.full_tensorboard_log = full_tensorboard_log

        #self.learning_rate_ph = None
        self.LR_ALPHA = None
        self.LR_BETA = None
        self.n_batch = None
        self.ADV_EX = None
        self.RET_EX = None
        self.R_EX = None
        self.DIS_V_MIX_LAST = None
        self.V_MIX = None
        self.A = None
        #self.actions_ph = None
        #A = tf.compat.v1.placeholder(tf.int32, [self.nbatch], 'A')
        #self.advs_ph = None
        #ADV_EX = tf.compat.v1.placeholder(tf.float32, [nbatch], 'ADV_EX')
        #self.rewards_ph = None
        #R_EX = tf.compat.v1.placeholder(tf.float32, [nbatch], 'R_EX')
        #RET_EX = tf.compat.v1.placeholder(tf.float32, [nbatch], 'RET_EX')
        #V_MIX = tf.compat.v1.placeholder(tf.float32, [nbatch], 'V_MIX')
        #DIS_V_MIX_LAST = tf.compat.v1.placeholder(tf.float32, [nbatch], 'DIS_V_MIX_LAST')
        #COEF_MAT = tf.compat.v1.placeholder(tf.float32, [nbatch, nbatch], 'COEF_MAT')  WE DON'T NEED COEF_MAT AND MATMUL(COEF_MAT * REWARDS), REWARD WOULD BE ENOUGH
        #LR_ALPHA = tf.compat.v1.placeholder(tf.float32, [], 'LR_ALPHA')
        #LR_BETA = tf.compat.v1.placeholder(tf.float32, [], 'LR_BETA')
        self.pg_mix_loss = None
        self.pg_ex_loss = None
        self.v_mix_loss = None
        self.v_ex_loss = None
        self.entropy = None
        #self.apply_backprop = None
        self.train_model = None
        self.step_model = None
        self.proba_step = None
        self.value = None
        self.initial_state = None
        self.learning_rate_schedule = None
        self.summary = None

        ### super(): 1. Allows us to avoid using the base class name explicitly
        ### 2. Working with Multiple Inheritance
        super(A2C, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True,
                                  _init_setup_model=_init_setup_model, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        # if we are loading, it is possible the environment is not known, however the obs and action space are known
        if _init_setup_model:
            self.setup_model()

    def _make_runner(self) -> AbstractEnvRunner:
        return A2CRunner(self.env, self, n_steps=self.n_steps, gamma=self.gamma)

    def _get_pretrain_placeholders(self):  #not used
        policy = self.train_model
        if isinstance(self.action_space, gym.spaces.Discrete):
            return policy.X, self.A, policy.policy
        return policy.X, self.A, policy.deterministic_action

    def setup_model(self):    # Part of the init in LIRPG A2C
        with SetVerbosity(self.verbose):
            # check if the input policy is in the class of A2C policies
            #assert issubclass(self.policy, ActorCriticPolicy), "Error: the input policy for the A2C model must be an " \
            #                                                    "instance of common.policies.ActorCriticPolicy."

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)    # returns a session that will use <num_cpu> CPU's only
                self.n_batch = self.n_envs * self.n_steps

                #line 55-56: Create step and train models
                n_batch_step = None
                n_batch_train = None
                if issubclass(self.policy, RecurrentActorCriticPolicy):
                    n_batch_step = self.n_envs
                    n_batch_train = self.n_envs * self.n_steps

                step_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs, 1,reuse=False)
                                         #n_batch_step, reuse=False, **self.policy_kwargs)
                # A context manager for defining ops that creates variables (layers).
                with tf.compat.v1.variable_scope("train_model", reuse=True,
                                       custom_getter=tf_util.outer_scope_getter("train_model")):
                    train_model = self.policy(self.sess, self.observation_space, self.action_space, self.n_envs*self.n_steps, self.n_steps, reuse=True)
                                              #self.n_steps, n_batch_train, reuse=True, **self.policy_kwargs)

                self.R_EX = tf.compat.v1.placeholder(tf.float32, [None], 'R_EX')
                self.DIS_V_MIX_LAST = tf.compat.v1.placeholder(tf.float32, [self.n_batch], 'DIS_V_MIX_LAST')
                self.COEF_MAT = tf.compat.v1.placeholder(tf.float32, [self.n_batch, self.n_batch], 'COEF_MAT')
                self.V_MIX = tf.compat.v1.placeholder(tf.float32, [None], 'V_MIX')
                self.A = tf.compat.v1.placeholder(tf.float32, [None, None], 'A')
                #nact = self.action_space.n
                #r_mix = self.r_ex_coef * self.R_EX + self.r_in_coef * tf.reduce_sum(train_model.r_in * tf.one_hot(self.A, nact), axis=1)
                r_mix = self.r_ex_coef * self.R_EX + self.r_in_coef * train_model.r_in
                #print("dimensions:", train_model.r_in, self.A)
                ret_mix = tf.squeeze(tf.matmul(self.COEF_MAT, tf.reshape(r_mix, [self.n_batch, 1])), [1]) + self.DIS_V_MIX_LAST
                adv_mix = ret_mix - self.V_MIX

                with tf.compat.v1.variable_scope("loss", reuse=False): #A context manager for defining ops that creates variables (layers).
                    #line 59-62: modify ex only to mix
                    #self.actions_ph = train_model.pdtype.sample_placeholder([None], name="action_ph")
                    #self.advs_ph = tf.compat.v1.placeholder(tf.float32, [None], name="advs_ph")
                    #self.rewards_ph = tf.compat.v1.placeholder(tf.float32, [None], name="rewards_ph")
                    #self.learning_rate_ph = tf.compat.v1.placeholder(tf.float32, [], name="learning_rate_ph")
                    self.LR_ALPHA = tf.compat.v1.placeholder(tf.float32, [], name="LR_ALPHA")
                    self.LR_BETA = tf.compat.v1.placeholder(tf.float32, [], name="LR_BETA")

                    #line 64-74: calculate loss
                    #print("@inside, A", self.A)
                    neglogpac = train_model.pd.neglogp(self.A)    #train_model.pd.neglogp in LIRPG
                    #self.entropy = tf.reduce_mean(train_model.pd.entropy())   #check if the ent formula same with LIRPG cat_entropy(train_model.pi)?
                    self.entropy = tf.reduce_mean(calc_entropy(train_model.pi)) #error location
                    self.pg_mix_loss = tf.reduce_mean(adv_mix * neglogpac)
                    self.v_mix_loss = tf.reduce_mean(mse(tf.squeeze(train_model.v_mix), ret_mix))
                    #rewards_ph is ret in LIRPGa
                    # https://arxiv.org/pdf/1708.04782.pdf#page=9, https://arxiv.org/pdf/1602.01783.pdf#page=4
                    # and https://github.com/dennybritz/reinforcement-learning/issues/34
                    # suggest to add an entropy component in order to improve exploration.
                    # Calculate the loss
                    # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss
                    # Policy loss
                    # L = A(s,a) * -logpi(a|s)

                    #loss = self.pg_loss - self.entropy * self.ent_coef + self.vf_loss * self.vf_coef
                    policy_loss = self.pg_mix_loss - self.ent_coef * self.entropy + self.v_mix_coef * self.v_mix_loss


                    # record in tf.summary for results printed
                    tf.summary.scalar('entropy_loss', self.entropy)
                    tf.summary.scalar('policy_gradient_loss', self.pg_mix_loss)
                    tf.summary.scalar('value_function_loss', self.v_mix_loss)
                    tf.summary.scalar('loss', policy_loss)

                    # Update parameters using loss (policy params update in LIRPG)
                    #line 77-91: cal grad and train (train is to update)
                    # 1. Get the model parameters
                    self.params = tf_util.get_trainable_vars("policy")

                    # 2. Calculate the gradients
                    grads = tf.gradients(policy_loss, self.params)
                    if self.max_grad_norm is not None:  # max_grad_norm defines the maximum gradient, needs to be normalized
                        # Clip the gradients (normalize)
                        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
                    grads_and_vars = list(zip(grads, self.params))  # zip pg and policy params correspondingly, policy_grads_and_vars in LIRPG

                with tf.compat.v1.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('discounted_rewards', tf.reduce_mean(ret_mix))
                    #tf.summary.histogram('learning_rate', self.learning_rate_ph)
                    tf.summary.scalar('learning_rate_alpha', tf.reduce_mean(self.LR_ALPHA))
                    tf.summary.scalar('learning_rate_beta', tf.reduce_mean(self.LR_BETA))
                    tf.summary.scalar('advantage', tf.reduce_mean(adv_mix))
                    if self.full_tensorboard_log:
                        tf.summary.histogram('discounted_rewards', self.ret_mix)
                        #tf.summary.histogram('learning_rate', self.learning_rate_ph)
                        tf.summary.scalar('learning_rate_alpha', tf.reduce_mean(self.LR_ALPHA))
                        tf.summary.scalar('learning_rate_beta', tf.reduce_mean(self.LR_BETA))
                        tf.summary.histogram('advantage', adv_mix)
                        if tf_util.is_image(self.observation_space):
                            tf.summary.image('observation', train_model.X)
                        else:
                            tf.summary.histogram('observation', train_model.X)
                # 3. Make up for one policy and value update step of A2C
                trainer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=self.LR_ALPHA, decay=self.alpha,
                                                    epsilon=self.epsilon, momentum=self.momentum)
                self.policy_train = trainer.apply_gradients(grads_and_vars)  #policy_train = policy_trainer.apply_gradients(policy_grads_and_vars)
                #print('policy_train:', self.policy_train)

                rmss = [trainer.get_slot(var, 'rms') for var in self.params]   ###NEED TO CHECK IF HAVE GET_SLOT()
                self.params_new = {}
                # Note: RMS stands for root mean square
                for grad, rms, var in zip(grads, rmss, self.params):
                    ms = rms + (tf.square(grad) - rms) * (1 - self.alpha) # wrong in this line
                    self.params_new[var.name] = var - self.LR_ALPHA * grad / tf.sqrt(ms + self.epsilon)  
                    #NEED TO CHECK IF HAVE .NAME

                # self.params_new assignment is problematic;
                self.policy_new = None
                self.policy_new = train_model.policy_new_fn(self.params_new, self.observation_space, self.action_space, n_batch_train, self.n_steps)
                ###NEED TO MODIFY POLICY -> HAVING CnnPolicyNew

                #start of intrinsic module updates
                self.ADV_EX = tf.compat.v1.placeholder(tf.float32, [None], 'ADV_EX')
                self.RET_EX = tf.compat.v1.placeholder(tf.float32, [None], 'RET_EX')
                neglogpac_new = self.policy_new.pd.neglogp(self.A)
                ratio_new = tf.exp(tf.stop_gradient(neglogpac) - neglogpac_new)
                self.pg_ex_loss = tf.reduce_mean(-self.ADV_EX * ratio_new)
                self.v_ex_loss = tf.reduce_mean(mse(tf.squeeze(train_model.v_ex), self.RET_EX))
                intrinsic_loss = self.pg_ex_loss + self.v_ex_coef * self.v_ex_loss

                print("save intrinsic loss:", intrinsic_loss)
                tf.print("tensors intrinsic loss:", intrinsic_loss, output_stream = sys.stderr)

                # record in tf.summary for results printed
                tf.summary.scalar('policy_gradient_loss', self.pg_ex_loss)
                tf.summary.scalar('value_function_loss', self.v_ex_loss)
                tf.summary.scalar('loss', intrinsic_loss)

                intrinsic_params = train_model.intrinsic_params  #CHECK IF HAVE .INTRINSIC_PARAMS
                intrinsic_grads = tf.gradients(intrinsic_loss, intrinsic_params)
                if self.max_grad_norm is not None:
                    intrinsic_grads, intrinsic_grad_norm = tf.clip_by_global_norm(intrinsic_grads, self.max_grad_norm)
                intrinsic_grads_and_vars = list(zip(intrinsic_grads, intrinsic_params))
                intrinsic_trainer = tf.optimizers.RMSprop(learning_rate=self.LR_BETA, decay=self.alpha, 
                                                          epsilon=self.epsilon, momentum=self.momentum)

                self.intrinsic_train = intrinsic_trainer.apply_gradients(intrinsic_grads_and_vars)
                #print('intr_train:', self.intrinsic_train)

                # line 150-159
                self.train_model = train_model
                self.step_model = step_model
                self.step = step_model.step
                #self.proba_step = step_model.proba_step  ### ENSURE SELF.PROBA_STEP CAN BE REMOVED
                self.value = step_model.value   ###self.model.value OR self.model.intrinsic_reward etc. are used in runner; model <- policy
                self.intrinsic_reward = step_model.intrinsic_reward
                self.initial_state = step_model.initial_state
                tf.compat.v1.global_variables_initializer().run(session=self.sess)

                self.summary = tf.compat.v1.summary.merge_all()

    def _train_step(self, obs, obs_nx, states, masks, actions, r_ex, ret_ex, v_ex, v_mix, dis_v_mix_last, coef_mat, update, writer=None):
        """
        applies a training step to the model

        :param obs: ([float]) The input observations
        :param states: ([float]) The states (used for recurrent policies)
        :param rewards: ([float]) The rewards from the environment
        :param masks: ([bool]) Whether or not the episode is over (used for recurrent policies)
        :param actions: ([float]) The actions taken
        :param values: ([float]) The logits values
        :param update: (int) the current step iteration
        :param writer: (TensorFlow Summary.writer) the writer for tensorboard
        :return: (float, float, float) policy loss, value loss, policy entropy
        """
        #line 120-136: train() in LIRPG, make the training part (feedforward and retropropagation of gradients)
        #advs = rewards - values
        advs_ex = ret_ex - v_ex
        #cur_lr = None
        for _ in range(len(obs)):
            #cur_lr = self.learning_rate_schedule.value()
            cur_lr_alpha = self.lr_alpha.value()
            cur_lr_beta = self.lr_beta.value()
        assert cur_lr_alpha is not None, "Error: the observation input array cannon be empty"
        assert cur_lr_beta is not None, "Error: the observation input array cannon be empty"
        
        ###CODE TO ADD TRAIN_MODEL.X_NEXT
        #TO AVOID A_ALL ERRORS, ADD INPUT IN {} OF POLICY
        td_map = {self.train_model.X: obs, self.policy_new.X:obs, self.A: actions, self.train_model.A_ALL: actions, self.train_model.X_NX: obs_nx, self.ADV_EX: advs_ex, self.RET_EX:ret_ex,
                  self.R_EX: r_ex, self.V_MIX:v_mix, self.DIS_V_MIX_LAST:dis_v_mix_last, self.COEF_MAT:coef_mat,
                  self.LR_ALPHA:cur_lr_alpha, self.LR_BETA:cur_lr_beta}

        if states is not None:
            td_map[self.train_model.states_ph] = states
            td_map[self.train_model.dones_ph] = masks

        #return self.sess.run([self.pg_mix_loss, self.entropy, self.policy_train, self.intrinsic_train], td_map)[0]
        #using td_map to compute, then fill in all the input of the previous 3 functions
        ### Using tf.summary.writer, else is the same; Is it same as save/load() in the baselines/a2c.py?
        ### The tf.summary module provides APIs for writing summary data. This data can be visualized in TensorBoard, the visualization toolkit that comes with TensorFlow.
        if writer is not None:
            # run loss backprop with summary, but once every 10 runs save the metadata (memory, compute time, ...)
            if self.full_tensorboard_log and (1 + update) % 10 == 0:

                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, pg_mix_loss, value_loss, policy_entropy, _, _ = self.sess.run(
                    [self.summary, self.pg_mix_loss, self.v_mix_loss, self.entropy, self.policy_train, self.intrinsic_train],
                    td_map, options=run_options, run_metadata=run_metadata)
                writer.add_run_metadata(run_metadata, 'step%d' % (update * self.n_batch))
            else:

                summary, pg_mix_loss, value_loss, policy_entropy, _ , _= self.sess.run(
                    [self.summary, self.pg_mix_loss, self.v_mix_loss, self.entropy, self.policy_train, self.intrinsic_train], td_map)
            writer.add_summary(summary, update * self.n_batch)

        else:
            pg_ex_loss, pg_mix_loss, value_mix_loss, value_ex_loss, policy_entropy, _, _ = self.sess.run(
                [self.pg_ex_loss, self.pg_mix_loss, self.v_mix_loss, self.v_ex_loss, self.entropy, self.policy_train, self.intrinsic_train], td_map)

        #print('TRY PG MIX:', pg_mix_loss)
        #print('TRY PG EX:', pg_ex_loss)

        return pg_ex_loss, pg_mix_loss, value_mix_loss, value_ex_loss, policy_entropy    #note: policy_entropy: entropy loss, policy_gradient_loss: pg_loss, value_function_loss, policy_loss; see line 224


### Very different learn function, consider use this learn()? or migrate to the baselines/a2c.py (have gamescoremean and v_mix_ev (explained varaince); just change the extrinsic performance evaluation?)
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="A2C",
              reset_num_timesteps=True):
        ## NO NEED TO HAVE SO MANY INPUT VALUES BECAUSE MODEL IS NOT INITIATED HERE??

        print("total_timesteps:", total_timesteps)
        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:
            self._setup_learn()
            #self.learning_rate_schedule = Scheduler(initial_value=self.learning_rate, n_values=total_timesteps,
            #                                        schedule=self.lr_schedule)
            self.lr_alpha = Scheduler(initial_value=self.lr_alpha, n_values=total_timesteps, schedule=self.lr_schedule)
            self.lr_beta = Scheduler(initial_value=self.lr_beta, n_values=total_timesteps, schedule=self.lr_schedule)

            t_start = time.time()
            self.ep_info_buf = deque(maxlen=100)
            self.eprexbuf = deque(maxlen=100)
            self.eprinbuf = deque(maxlen=100)
            self.eplenbuf = deque(maxlen=100)
            callback.on_training_start(locals(), globals())

            for update in range(1, total_timesteps // self.n_batch + 1):
                # Get mini batch of experiences
                callback.on_rollout_start()
                # true_reward is the reward without discount
                rollout = self.runner.run(callback)
                # unpack
                #obs, actions, states, rewards, masks,  values, ep_infos, true_reward = rollout
                
                ###CODE TO ADD IN OBS_NX
                obs, actions, obs_nx, states, r_in, r_ex, ret_ex, ret_mix, \
                v_ex, v_mix, last_v_ex, last_v_mix, masks, dones, \
                ep_info, ep_r_ex, ep_r_in, ep_len = rollout


                dis_v_mix_last = np.zeros([self.n_batch], np.float32)
                coef_mat = np.zeros([self.n_batch, self.n_batch], np.float32)
                for i in range(self.n_batch):
                    dis_v_mix_last[i] = self.gamma ** (self.n_steps - i % self.n_steps) * last_v_mix[i // self.n_steps]
                    coef = 1.0
                    for j in range(i, self.n_batch):
                        if j > i and j % self.n_steps == 0:
                            break
                        coef_mat[i][j] = coef
                        coef *= self.gamma
                        if dones[j]:
                            dis_v_mix_last[i] = 0
                            break
                callback.update_locals(locals())
                callback.on_rollout_end()

                # Early stopping due to the callback
                if not self.runner.continue_training:
                    break

                self.ep_info_buf.extend(ep_info)
                self.eprexbuf.extend(ep_r_ex)
                self.eprinbuf.extend(ep_r_in)
                self.eplenbuf.extend(ep_len)

                pg_ex_loss, pg_mix_loss, value_mix_loss, value_ex_loss, policy_entropy = self._train_step(obs, obs_nx, states, masks, actions, r_ex, ret_ex, v_ex, v_mix,
                                                                 dis_v_mix_last, coef_mat, self.num_timesteps // self.n_batch, writer)
                policy_loss = pg_mix_loss - self.ent_coef * policy_entropy + self.v_mix_coef * value_mix_loss
                intrinsic_loss = pg_ex_loss + self.v_ex_coef * value_ex_loss
                #print("pg_mix_loss in for:", pg_mix_loss)
                #print("pg_ex_loss in for:", pg_ex_loss)
                f = open("C:/Users/Zihang Guan/Desktop/Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020-master/results/convergence/loss.csv", 'a', newline = '')
                to_append = [[update, pg_mix_loss, pg_ex_loss, value_mix_loss, value_ex_loss, policy_entropy, policy_loss, intrinsic_loss]]
                csvwriter = csv.writer(f)
                csvwriter.writerows(to_append)
                f.close()
                
                n_seconds = time.time() - t_start
                # Calculate the fps (frame per second)
                fps = int((update * self.n_batch) / n_seconds)

                if writer is not None:
                    total_episode_reward_logger(self.episode_reward,
                                                true_reward.reshape((self.n_envs, self.n_steps)),
                                                masks.reshape((self.n_envs, self.n_steps)),
                                                writer, self.num_timesteps)

                if self.verbose >= 1 and (update % log_interval == 0 or update == 1):
                    logger.record_tabular("nupdates", update)
                    logger.record_tabular("total_timesteps", self.num_timesteps)
                    logger.record_tabular("fps", fps)
                    logger.record_tabular("policy_entropy", float(policy_entropy))
                    #logger.record_tabular("value_loss", float(value_loss))
                    logger.record_tabular("v_ex_loss", float(v_ex_loss))
                    logger.record_tabular("v_mix_loss", float(v_mix_loss))
                    #explained_var = explained_variance(values, rewards)
                    v_ex_ev = explained_variance(v_ex, ret_ex)
                    logger.record_tabular("v_ex_ev", float(v_ex_ev))
                    v_mix_ev = explained_variance(v_mix, ret_mix)
                    logger.record_tabular("v_mix_ev", float(v_mix_ev))
                    logger.record_tabular("explained_variance", float(explained_var))
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_reward_mean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('ep_len_mean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.dump_tabular()

        callback.on_training_end()
        return self

    def save(self, save_path, cloudpickle=False):
        data = {
            "gamma": self.gamma,
            "n_steps": self.n_steps,
            #"vf_coef": self.vf_coef,
            "v_mix_coef": self.v_mix_coef,
            "v_ex_coef": self.v_ex_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            #"learning_rate": self.learning_rate,
            "lr_alpha": self.lr_alpha,
            "lr_beta": self.lr_beta,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "lr_schedule": self.lr_schedule,
            "verbose": self.verbose,
            "policy": self.policy,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }
        #line 138-141: save() in LIRPG, save the model
        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)


class A2CRunner(AbstractEnvRunner):
    def __init__(self, env, model, n_steps=5, gamma=0.99, r_ex_coef=1.0, r_in_coef=0.01):
        """
        A runner to learn the policy of an environment for an a2c model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        """
        #line 162-176: needs to modify the parameters in runner; MARKED LINES TO TAKE CARE LATER
        super(A2CRunner, self).__init__(env=env, model=model, n_steps=n_steps)
        self.gamma = gamma
        self.r_ex_coef = r_ex_coef
        self.r_in_coef = r_in_coef

        #self.batch_ob_shape = (self.n_envs*self.n_steps,) + env.observation_space.shape
        #self.obs = env.reset()
        #self.policy_states = model.initial_state
        #self.dones = [False for _ in range(self.n_envs)]
        nenv = env.num_envs
        self.ep_r_in = np.zeros([nenv])
        self.ep_r_ex = np.zeros([nenv])
        self.ep_len = np.zeros([nenv])

    def _run(self):
        """
        Run a learning step of the model

        :return: ([float], [float], [float], [bool], [float], [float])
                 observations, states, rewards, masks, actions, values
        """
        #mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        mb_obs, mb_r_ex, mb_r_in, mb_actions, mb_v_ex, mb_v_mix, mb_dones = [],[],[],[],[],[],[]
        mb_obs_next = []
        mb_states = self.states
        #ep_infos = []
        ep_info, ep_r_ex, ep_r_in, ep_len = [], [], [], []
        for _ in range(self.n_steps):
            actions, v_ex, v_mix, states, _ = self.model.step(self.obs, self.states, self.dones)  # pytype: disable=attribute-error
            mb_obs.append(np.copy(self.obs))
            #print('mb_obs:', self.obs)
            mb_actions.append(actions)
            #mb_values.append(values)
            mb_v_ex.append(v_ex)
            mb_v_mix.append(v_mix)
            mb_dones.append(self.dones)
            clipped_actions = actions

            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            obs, r_ex, dones, infos = self.env.step(clipped_actions)
            mb_obs_next.append(obs)
            #print('mb_obs_nx:', obs)
            if mb_obs is None:
                r_in = self.model.intrinsic_reward(self.obs, clipped_actions)
            else: 
                r_in = self.model.intrinsic_reward(self.obs, clipped_actions, obs)    ###EXPAND TO INCULDE OBS_NX, obs = OBS_NX, self.obs = OBS
            mb_r_ex.append(r_ex)
            mb_r_in.append(r_in)

            self.model.num_timesteps += self.n_envs    # update of step counter t

            if self.callback is not None:
                # Abort training early
                self.callback.update_locals(locals())
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 8

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_info.append(maybe_ep_info)

            self.states = states
            self.dones = dones
            self.obs = obs
            self.ep_r_ex += r_ex
            self.ep_r_in += r_in
            self.ep_len += 1

            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
                    ep_r_ex.append(self.ep_r_ex[n])
                    ep_r_in.append(self.ep_r_in[n])
                    ep_len.append(self.ep_len[n])
                    self.ep_r_ex[n], self.ep_r_in[n], self.ep_len[n] = 0,0,0

        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_obs_nx = np.asarray(mb_obs_next, dtype=self.obs.dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        ###CODE FOR MB_OBS_NX
        mb_r_ex = np.asarray(mb_r_ex, dtype=np.float32).swapaxes(1, 0)
        mb_r_in = np.asarray(mb_r_in, dtype=np.float32).swapaxes(1, 0)
        #print("check intrinsic is learning:", mb_r_in)

        mb_r_mix = self.r_ex_coef * mb_r_ex + self.r_in_coef * mb_r_in
        #print("check how much intrinsic is learning:", mb_r_in / mb_r_mix)
        #THE ABOVE 3 LINES SHOULD REPLACE THE BELOW LINE to incorporate an intrinsic reward
        #mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
        mb_actions = np.asarray(mb_actions, dtype=self.env.action_space.dtype).swapaxes(0, 1)
        #mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(0, 1)
        mb_v_ex = np.asarray(mb_v_ex, dtype=np.float32).swapaxes(1, 0)
        mb_v_mix = np.asarray(mb_v_mix, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        #true_rewards = np.copy(mb_rewards)
        #last_values = self.model.value(self.obs, self.states, self.dones).tolist()  # pytype: disable=attribute-error
        last_v_ex, last_v_mix = self.model.value(self.obs, self.states, self.dones)
        last_v_ex, last_v_mix = last_v_ex.tolist(), last_v_mix.tolist()
        mb_ret_ex, mb_ret_mix = np.zeros(mb_r_ex.shape), np.zeros(mb_r_mix.shape)
        # discount/bootstrap off value fn
        #for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
        for n, (r_ex, r_mix, dones, v_ex, v_mix) in enumerate(zip(mb_r_ex, mb_r_mix, mb_dones, last_v_ex, last_v_mix)):
            #rewards = rewards.tolist()
            r_ex, r_mix = r_ex.tolist(), r_mix.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                #rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
                ret_ex = discount_with_dones(r_ex + [v_ex], dones + [0], self.gamma)[:-1]
                ret_mix = discount_with_dones(r_mix + [v_mix], dones + [0], self.gamma)[:-1]
            else:
                #rewards = discount_with_dones(rewards, dones, self.gamma)
                ret_ex = discount_with_dones(r_ex, dones, self.gamma)
                ret_mix = discount_with_dones(r_mix, dones, self.gamma)
            #mb_rewards[n] = rewards
            mb_ret_ex[n], mb_ret_mix[n] = ret_ex, ret_mix

        # convert from [n_env, n_steps, ...] to [n_steps * n_env, ...]
        #mb_rewards = mb_rewards.reshape(-1, *mb_rewards.shape[2:])
        mb_r_ex = mb_r_ex.flatten()
        mb_r_in = mb_r_in.flatten()
        mb_ret_ex = mb_ret_ex.flatten()
        mb_ret_mix = mb_ret_mix.flatten()
        mb_actions = mb_actions.reshape(-1, *mb_actions.shape[2:])
        #mb_values = mb_values.reshape(-1, *mb_values.shape[2:])
        mb_v_ex = mb_v_ex.reshape(-1, *mb_v_ex.shape[2:])
        mb_v_mix = mb_v_mix.reshape(-1, *mb_v_mix.shape[2:])
        mb_masks = mb_masks.reshape(-1, *mb_masks.shape[2:])
        mb_dones = mb_dones.flatten()
        #true_rewards = true_rewards.reshape(-1, *true_rewards.shape[2:])
        #return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, ep_infos, true_rewards

        return mb_obs, mb_actions, mb_obs_nx, mb_states,mb_r_in, mb_r_ex, mb_ret_ex, mb_ret_mix, \
               mb_v_ex, mb_v_mix, last_v_ex, last_v_mix, mb_masks, mb_dones, \
               ep_info, ep_r_ex, ep_r_in, ep_len
               #true_rewards
'''import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', help='Environment ID', default='BreakoutNoFrameskip-v4'), use env_train instead
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    #parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'cnn_int'],
    #                    default='cnn_int'), policies for each algorithm is specified out already, may consider to modify to intrinsic augmented policy
    #parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='linear'), sepcified as 'constant' in models (a2c etc.)
    #parser.add_argument('--num-timesteps', type=int, default=int(50E6)), alr specified as input to each algorithm
    parser.add_argument('--v-ex-coef', type=float, default=0.1)
    parser.add_argument('--r-ex-coef', type=float, default=1)
    parser.add_argument('--r-in-coef', type=float, default=0.01)
    #parser.add_argument('--lr-alpha', type=float, default=7E-4)
    #parser.add_argument('--lr-beta', type=float, default=7E-4)
    args = parser.parse_args()
    logger.configure()
    train( seed=args.seed,
          v_ex_coef=args.v_ex_coef, r_ex_coef=args.r_ex_coef, r_in_coef=args.r_in_coef)'''''