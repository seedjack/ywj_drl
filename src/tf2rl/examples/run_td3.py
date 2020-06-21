# import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import gym
import rospy
from gazebo_env.environment_stage_3 import Env
from tf2rl.algos.td3 import TD3
from tf2rl.experiments.trainer import Trainer


Load = True
if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = TD3.get_argument(parser)
    # parser.add_argument('--env-name', type=str, default="2dCarWorld2-v1")
    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=10000) # 重新训练的话要改回 10000
    parser.set_defaults(max_steps=2e5)
     # restore 静态训练的结果去训练动态，看看效果如何
    # parser.set_defaults(
    #     model_dir='/home/ywj/ywj_ws/src/tf2rl/examples/results/sac_train_evn/20200515T145317.384540_TD3_')

    if Load:
        parser.set_defaults(test_episodes=100)
        parser.set_defaults(episode_max_steps=int(1e4))
        parser.set_defaults(model_dir='/home/ywj/ywj_ws/src/tf2rl/examples/results/test/20200518T091228.757412_TD3_')
        # parser.set_defaults(model_dir='/home/ywj/gym_ws/tf2rl/examples/results/test_500k_td3/20200311T005309.078819_TD3_')
        parser.set_defaults(show_test_progress=False)
        parser.set_defaults(save_model_interval=int(1e10))
    args = parser.parse_args()

    # env = gym.make(args.env_name)
    # test_env = gym.make(args.env_name)
    rospy.init_node('turtlebot3_td3_stage_1')
    env = Env()
    test_env = Env()

    policy = TD3(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=-1,
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        actor_units=[400, 300],
        n_warmup=args.n_warmup)
    trainer = Trainer(policy, env, args, test_env=test_env)
    if Load:
        trainer.evaluate_policy(10000)  # 每次测试都会在生成临时文件，要定期处理
    else:
        trainer()
