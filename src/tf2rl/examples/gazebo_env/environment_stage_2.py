import rospy
import numpy as np
import math
from gym import spaces
from gym.utils import seeding
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
# from tf.transformations import euler_from_quaternion, quaternion_from_euler
from .respawnGoal import Respawn


class Env:
    def __init__(self):
        self.goal_x = 1.0
        self.goal_y = 0
        self.inflation_rad = 0.25
        self.heading = 0
        self.pre_heading = 0
        self.max_v = 0.2
        self.max_w = 1.5
        self.goal_threshold = 0.15
        self.collision_threshold = 0.17
        self.vel_cmd = [0., 0.]
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.num_beams = 10  # 激光数
        low = np.array([-1.5])
        high = np.array([1.5])
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        low = [0.0] * (self.num_beams)
        low.extend([0., -1.5, -2*pi, 0])  #极坐标
        # low.extend([0., -1.5, -2.0, -2.0,-2.0, -2.0]) #笛卡尔坐标
        high = [3.5] * (self.num_beams)
        high.extend([0.2, 1.5, 2*pi, 4])
        # high.extend([0.2, 1.5, 2.0, 2.0, 2.0, 2.0])
        self.observation_space = spaces.Box(np.array(low), np.array(high), dtype=np.float32)

        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()

    def euler_from_quaternion(self, orientation_list):
        x, y, z, w = orientation_list
        r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        p = math.asin(2 * (w * y - z * x))
        y = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))

        return r, p, y


    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = self.euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self, scan):
        scan_range = []
        heading = self.heading
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        # obstacle_min_range = round(min(scan_range), 2)
        # obstacle_angle = np.argmin(scan_range)
        if self.collision_threshold > min(scan_range) > 0:
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        if current_distance < self.goal_threshold:
            self.get_goalbox = True

        state = scan_range + self.vel_cmd + [heading, current_distance] # 极坐标
        # state = scan_range + self.vel_cmd + [self.position.x, self.position.y, self.goal_x, self.goal_y] #笛卡尔坐标
        return state, done

    def setReward(self, state, done):
        # yaw_reward = []
        # current_distance = state[-1]
        # heading = state[-2]

        # for i in range(5):
        #     angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
        #     tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
        #     yaw_reward.append(tr)
        #
        # distance_rate = 2 ** (current_distance / self.goal_distance)
        # reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate)

        if done:
            rospy.loginfo("Collision!!")
            reward = -150
            self.pub_cmd_vel.publish(Twist())

        elif self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 200
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True, test=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False
        else:
            reward = (self.goal_threshold-state[-1])/4.0

        # # 增加一层膨胀区域，越靠近障碍物负分越多
        # obstacle_min_range = round(min(state[:10]), 2)
        # if obstacle_min_range < self.inflation_rad:
        #     reward -= 100*(1 - obstacle_min_range/self.inflation_rad)

        return reward

    def seed(self, seed=None):
        # 产生一个随机化时需要的种子，同时返回一个np_random对象，支持后续的随机化生成操作
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.pre_heading = self.heading
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = action
        self.vel_cmd = [vel_cmd.linear.x, vel_cmd.angular.z]
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward = self.setReward(state, done)

        # 到达目标或者碰撞到障碍物都reset
        return np.array(state), reward, done or reward==200, {}

    def render(self):
        pass

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.vel_cmd = [0., 0.]
        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        return np.array(state)



