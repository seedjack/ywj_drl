# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# import gym
import rospy
from gazebo_env.environment_stage_2 import Env
from tf2rl.algos.sac_discrete import SACDiscrete
from tf2rl.experiments.trainer import Trainer

Load = False
if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = SACDiscrete.get_argument(parser)
    # parser.set_defaults(test_interval=2000)
    # parser.add_argument('--env-name', type=str,
    #                     default="2dCarWorld1-v1")
    parser.set_defaults(max_steps=5e5)
    parser.set_defaults(gpu=-1)
    parser.set_defaults(n_warmup=100)
    parser.set_defaults(batch_size=100)
    # parser.set_defaults(
    #     model_dir='/home/ywj/ywj_ws/src/tf2rl/examples/results/sac_train_evn/20200510T183903.408701_SAC_discrete_')
    if Load:
        parser.set_defaults(test_episodes=100)
        parser.set_defaults(episode_max_steps=int(1e4))
        parser.set_defaults(model_dir='/home/ywj/ywj_ws/src/tf2rl/examples/results/test/20200517T101732.502613_TD3_')
        # parser.set_defaults(model_dir='/home/ywj/gym_ws/tf2rl/examples/results/test_500k_td3/20200311T005309.078819_TD3_')
        parser.set_defaults(show_test_progress=True)
        parser.set_defaults(save_model_interval=int(1e10))
    args = parser.parse_args()

    rospy.init_node('turtlebot3_td3_stage_1')
    env = Env()
    test_env = Env()
    # env = gym.make(args.env_name)
    # test_env = gym.make(args.env_name)
    policy = SACDiscrete(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.n,
        discount=0.99,
        gpu=args.gpu,
        memory_capacity=args.memory_capacity,
        batch_size=args.batch_size,
        n_warmup=args.n_warmup)
    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
