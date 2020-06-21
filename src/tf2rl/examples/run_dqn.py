# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import gym
import rospy
from gazebo_env.environment_stage_2 import Env
from tf2rl.algos.dqn import DQN
from tf2rl.experiments.trainer import Trainer

if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = DQN.get_argument(parser)
    parser.set_defaults(test_interval=2000)
    parser.set_defaults(max_steps=2e3)
    parser.set_defaults(gpu=-1)
    parser.set_defaults(n_warmup=500)
    parser.set_defaults(batch_size=32)
    parser.set_defaults(memory_capacity=int(1e4))
    args = parser.parse_args()

    # env = gym.make("CartPole-v0")
    # test_env = gym.make("CartPole-v0")
    rospy.init_node('turtlebot3_dqn_stage_2_tensorflow')
    env = Env()
    test_env = Env()


    policy = DQN(
        enable_double_dqn=args.enable_double_dqn,
        enable_dueling_dqn=args.enable_dueling_dqn,
        enable_noisy_dqn=args.enable_noisy_dqn,
        enable_categorical_dqn=args.enable_categorical_dqn,
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        target_replace_interval=300,
        discount=0.99,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        batch_size=args.batch_size,
        n_warmup=args.n_warmup)
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
