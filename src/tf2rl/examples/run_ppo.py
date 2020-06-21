import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


import gym
from tf2rl.algos.ppo import PPO
from tf2rl.policies.categorical_actor import CategoricalActorCritic
from tf2rl.experiments.on_policy_trainer import OnPolicyTrainer
from tf2rl.envs.utils import is_discrete, get_act_dim
from gym_extensions.continuous import gym_navigation_2d

Load = False

if __name__ == '__main__':
    parser = OnPolicyTrainer.get_argument()
    parser = PPO.get_argument(parser)
    parser.add_argument('--env-name', type=str,
                        default="2dCarWorld1-v1")
    parser.set_defaults(test_interval=20480)
    parser.set_defaults(max_steps=int(5e5))
    parser.set_defaults(horizon=2048)
    parser.set_defaults(batch_size=64)
    parser.set_defaults(gpu=-1)

    if Load:
        parser.set_defaults(test_episodes=30)
        parser.set_defaults(model_dir='/home/ywj/gym_ws/tf2rl/examples/results/static_target/20191210T204318.737131_PPO_')
        parser.set_defaults(show_test_progress=True)
        parser.set_defaults(save_model_interval=1e10)

    args = parser.parse_args()

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    # env = gym.make("2dCarWorld2-v1")
    # test_env = gym.make("2dCarWorld2-v1")

    policy = PPO(
        state_shape=env.observation_space.shape,
        action_dim=get_act_dim(env.action_space),
        is_discrete=is_discrete(env.action_space),
        max_action=None if is_discrete(
            env.action_space) else env.action_space.high[0],
        batch_size=args.batch_size,
        actor_units=[64, 64],
        critic_units=[64, 64],
        n_epoch=10,
        n_epoch_critic=10,
        lr_actor=3e-4,
        lr_critic=3e-4,
        discount=0.99,
        lam=0.95,
        horizon=args.horizon,
        normalize_adv=args.normalize_adv,
        enable_gae=args.enable_gae,
        gpu=args.gpu)
    trainer = OnPolicyTrainer(policy, env, args, test_env=test_env)

    if Load:
        trainer.evaluate_policy(10000)  # 每次测试都会在result/test生成临时文件，要定期处理
    else:
        trainer()

