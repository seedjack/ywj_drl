import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams['font.family'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# continuous
# continuous_TD3 = pd.read_csv('/home/ywj/data/coordinate/run-cross_coordinates-tag-Common_training_return.csv')
# continuous_SAC = pd.read_csv('/home/ywj/data/coordinate/run-polar_coordinates_update_env_code-tag-Common_training_return.csv')
# continuous_PPO = pd.read_csv('/home/ywj/data/coordinate/run-single_polar_coodinates-tag-Common_training_return.csv')
continuous_TD3 = pd.read_csv('/home/ywj/下载/td3_500k.csv')
# continuous_SAC = pd.read_csv('/home/ywj/下载/inflation.csv')
# continuous_PPO = pd.read_csv('/home/ywj/下载/td3_500k.csv')
# continuous_DDPG = pd.read_csv('/home/ywj/data/coordinate')

continuous_data = [continuous_TD3]
smoothed_data = []
for y in continuous_data:
    smoothed = []
    weight = 0.99
    last = y['Value'].values[0]
    for point in y['Value'].values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    smoothed_data.append(smoothed)
# print(type(x), type(y), len(x), len(y))
plt.figure()
plt.plot(continuous_TD3['Step'].values, smoothed_data[0], 'k-.')
# # plt.plot(continuous_SAC['Step'].values, smoothed_data[1],  label='no_transfer')
# plt.plot(continuous_PPO['Step'].values, smoothed_data[2], 'g:')

# plt.plot(continuous_DDPG['Step'].values, smoothed_data[3], label='DDPG')
plt.xlabel('steps')
plt.ylabel('reward')
plt.legend(loc='upper left')
plt.grid(linestyle='-.')
plt.savefig('/home/ywj/test.png')
plt.show()
#
# # discrete
# # continuous_PPO = pd.read_csv('/home/ywj/data/discrete/run-20191213T003216.106968_PPO_-tag-Common_training_return.csv')
# # continuous_SAC = pd.read_csv('/home/ywj/data/discrete/run-20191213T103406.918472_SAC_discrete_-tag-Common_training_return.csv')
# # continuous_Rainbow = pd.read_csv('/home/ywj/data/discrete/run-20191213T231752.919630_D6QN_-tag-Common_training_return.csv')
# #
# # continuous_data = [continuous_PPO, continuous_SAC, continuous_Rainbow]
# # smoothed_data = []
# # for y in continuous_data:
# #     smoothed = []
# #     weight = 0.99
# #     last = y['Value'].values[0]
# #     for point in y['Value'].values:
# #         smoothed_val = last * weight + (1 - weight) * point
# #         smoothed.append(smoothed_val)
# #         last = smoothed_val
# #     smoothed_data.append(smoothed)
# # # print(type(x), type(y), len(x), len(y))
# # plt.figure()
# # plt.plot(continuous_PPO['Step'].values, smoothed_data[0], label='PPO')
# # plt.plot(continuous_SAC['Step'].values, smoothed_data[1], label='SAC')
# # plt.plot(continuous_Rainbow['Step'].values, smoothed_data[2], label='Rainbow')
# # plt.xlabel('Steps')
# # plt.ylabel('Value')
# # plt.legend(loc='upper left')
# # plt.grid(linestyle='-.')
# # plt.savefig('/home/ywj/data/discrete/discrete.jpg')
# # plt.show()

# # turtlebot
# continuous_PPO = pd.read_csv('/home/ywj/data/test_500k_td3/run-20200311T005309.078819_TD3_-tag-Common_training_return.csv')
# smoothed_data = []
#
# smoothed = []
# weight = 0.9975
# last = continuous_PPO['Value'].values[0]
# for point in continuous_PPO['Value'].values:
#     smoothed_val = last * weight + (1 - weight) * point
#     smoothed.append(smoothed_val)
#     last = smoothed_val
# smoothed_data.append(smoothed)
# # print(type(x), type(y), len(x), len(y))
# plt.figure()
# plt.plot(continuous_PPO['Step'].values, smoothed_data[0], label='TD3')
#
# plt.xlabel('步数')
# plt.ylabel('奖励/回合')
# plt.legend(loc='upper left')
# plt.grid(linestyle='-.')
# plt.savefig('/home/ywj/data/picture/test_500k_td3/test_500k_td3.jpg')
# plt.show()
