import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import gym

from tf2rl.algos.ddpg import DDPG
from tf2rl.experiments.trainer import Trainer

Load = False
if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = DDPG.get_argument(parser)
    parser.add_argument('--env-name', type=str, default="2dCarWorld2-v1")
    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=10000)
    if Load:
        parser.set_defaults(test_episodes=30)
        parser.set_defaults(model_dir='/home/ywj/gym_ws/tf2rl/examples/results/add_collision_to_PuckWorld/20191210T162040.076071_DDPG_')
        parser.set_defaults(show_test_progress=True)
        parser.set_defaults(save_model_interval=1e10)
    args = parser.parse_args()

    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    policy = DDPG(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=-1,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup)
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
