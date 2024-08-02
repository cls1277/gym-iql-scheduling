# Date: 2024/8/1 9:21
# Author: cls1277
# Email: cls1277@163.com

import gym
import numpy as np
from utils.config import Config
import random

state_length = 35
PMindex2name = {
    0: "Alignment",
    1: "Plasma1",
    2: "Plasma2",
    3: "Clean1",
    4: "Clean2",
    5: "AVM",
    6: "PreAligner",
    7: "WaferBuffer"
}
PENALTY = -200000
AWARD = 200000

class MachineEnv(gym.Env):
    def __init__(self):
        self.state = np.concatenate([-np.ones(6), np.zeros(16), -np.ones(10), [6], np.zeros(state_length - 33)]).astype(int)
        self.reward = 0
        self.done = False
        self.info = {}
        Config.load_config()
        self.jobs = Config.get_value("job_number")

    def reset(self):
        self.state = np.concatenate([-np.ones(6), np.zeros(16), -np.ones(10), [6], np.zeros(state_length - 33)]).astype(int)
        self.reward = 0
        self.done = False
        self.info = {}


    # 通过腔室的编号得到需要使用的片子的下标
    def _get_Wafer_index(self, PM_index):
        Wafer_indexs = np.where(self.state[0:6] == PM_index)[0]
        if Wafer_indexs.size != 1:
            return -1
        return Wafer_indexs[0]

    def _get_Wafer_location(self, Wafer_index):
        return self.state[Wafer_index]

    def _set_Wafer_location(self, Wafer_index, location):
        self.state[Wafer_index] = location

    def _get_Wafer_status(self, Wafer_index):
        return self.state[6 + Wafer_index]

    def _set_Wafer_status(self, Wafer_index, status):
        self.state[6 + Wafer_index] = status

    def _get_PM_status(self, PM_index):
        return self.state[12 + PM_index]

    def _set_PM_status(self, PM_index, status):
        self.state[12 + PM_index] = status

    def _set_PM_remain(self, PM_index, remain):
        self.state[22 + PM_index] = remain

    def _get_remain_time(self, PM_index):
        return self.state[22 + PM_index]

    def _get_remain_times(self):
        return self.state[22:32]

    def _update_remain_times(self, pass_time):
        self.state[22:32][self.state[22:32] != -1] -= pass_time

    def _get_VTM_facing(self):
        return self.state[32]

    def _set_VTM_facing(self, facing):
        self.state[32] = facing

    # 0 是左手, 1 是右手
    def _get_VTM_hands(self):
        return self.state[33:35]

    # 让手指翻转状态
    def _set_VTM_hands(self, hand):
        self.state[33 + hand] = 1 - self.state[33 + hand]

    # waiting 之后的新状态为 +
    def _after_waiting(self, PM_index):
        self.state[12 + PM_index] *= -1
        self._set_PM_remain(PM_index, Config.get_value(PMindex2name[PM_index] + "." + "duration")[self.state[12 + PM_index]])

    def _get_buffer_count(self):
        return len(np.where(self.state[0:6] == 7))

    def _check_facing_pick(self):
        PM_index = self._get_VTM_facing()
        PM_status = self._get_PM_status(PM_index)
        if (PM_index == 0 and PM_status == -9) or (1 <= PM_index <= 5 and PM_status == -6) or (PM_index == 6 and PM_status == -1) or (PM_index == 7 and self._get_buffer_count() > 0):
            return True
        else:
            return False

    def _check_facing_place(self):
        PM_index = self._get_VTM_facing()
        PM_status = self._get_PM_status(PM_index)
        if (PM_index == 0 and (PM_status == -2 or PM_status == -5)) or (1 <= PM_index <= 5 and PM_status == -2) or (PM_index == 6 and PM_status == 0) or (PM_index == 7 and self._get_buffer_count() < 4):
            return True
        else:
            return False

    # 更新每个腔室的状态
    def _update_status_0(self):
        for PM_index in range(7):
            # 剩余时间是 0 ,说明这个腔室要进入下一个状态了
            if self._get_remain_time(PM_index) != 0:
                continue
            if self._get_PM_status(PM_index) != 0:
                PM_status = self._get_PM_status(PM_index)
                # Alignment
                if PM_index == 0:
                    if PM_status == 1 or PM_status == 4 or PM_status == 8:
                        # 不会更新片子的位置和状态
                        # 用 -2 来表示刚刚从 1 状态结束,等待机械手操作进入 2 状态
                        self._set_PM_status(PM_index, -(PM_status + 1))
                        self._set_PM_remain(PM_index, -1)
                    else:
                        self._set_PM_status(PM_index, PM_status + 1)
                        self._set_PM_remain(PM_index, Config.get_value(PMindex2name[PM_index] + "." + "duration")[PM_status + 1])
                        # Alignment Process 做完
                        if PM_status == 7:
                            # 把 Alignment 上加工完的 Wafer 的状态设为 <Align后>
                            # self._set_Wafer_status(self._get_Wafer_index(PM_index), 5)
                            Wafer_index = self._get_Wafer_index(PM_index)
                            if Wafer_index == -1:
                                self.reward = PENALTY
                                return
                            self._set_Wafer_status(Wafer_index, self._get_Wafer_status(Wafer_index) + 1)
                # Plasma1, Plasma2, Clean1, Clean2, AVM
                elif 1 <= PM_index <= 5:
                    if PM_status == 1 or PM_status == 5:
                        self._set_PM_status(PM_index, -(PM_status + 1))
                        self._set_PM_remain(PM_index, -1)
                    else:
                        self._set_PM_status(PM_index, PM_status + 1)
                        self._set_PM_remain(PM_index, Config.get_value(PMindex2name[PM_index] + "." + "duration")[PM_status + 1])
                        # 这些腔室的 Process 做完
                        if PM_status == 4:
                            # PMindex2status = {
                            #     1: 2,
                            #     2: 2,
                            #     3: 3,
                            #     4: 3,
                            #     5: 6
                            # }
                            # self._set_Wafer_status(self._get_Wafer_index(PM_index), PMindex2status[PM_index])
                            Wafer_index = self._get_Wafer_index(PM_index)
                            if Wafer_index == -1:
                                self.reward = PENALTY
                                return
                            self._set_Wafer_status(Wafer_index, self._get_Wafer_status(Wafer_index) + 1)
                else:
                    if PM_status == 1:
                        self._set_PM_status(PM_index, -1)
                        self._set_PM_remain(PM_index, -1)
                        Wafer_index = self._get_Wafer_index(PM_index)
                        if Wafer_index == -1:
                            self.reward = PENALTY
                            return
                        self._set_Wafer_status(Wafer_index, self._get_Wafer_status(Wafer_index) + 1)
            else:
                print("Error: wrong PM_status: " + str(PM_index) + " " + str(self._get_PM_status(PM_index)))

    def _update_status_VTM_1234(self, do_type, hand):
        self._set_VTM_hands(hand)
        self._after_waiting(self._get_VTM_facing())
        Wafer_index = self._get_Wafer_index(self._get_VTM_facing())
        if Wafer_index == -1:
            self.reward = PENALTY
            return
        if do_type == "pick":
            self._set_Wafer_location(Wafer_index, 8 + hand)
        elif do_type == "place":

            self._set_Wafer_location(Wafer_index, self._get_VTM_facing())
        else:
            print("Error: do_type is not pick or place")


    def step(self, action, agent_index):

        # VTM
        if agent_index == 0:
            if action == 0:
                remain_times = self._get_remain_times()
                remain_times_normal = remain_times[remain_times != -1]
                if len(remain_times_normal) == 0:
                    self.reward = PENALTY
                    return
                min_remain_time = np.min(remain_times_normal)
                delta_time = random.randint(min_remain_time // 2, min_remain_time)
                self._update_remain_times(delta_time)
                self._update_status_0()
                self.reward = -delta_time
            elif action == 1:
                if self._check_facing_pick() and self._get_VTM_hands()[1] == 0:
                    self._update_status_VTM_1234("pick", 1)
                    self.reward = 0
                else:
                    self.reward = PENALTY
            elif action == 2:
                if self._check_facing_place() and self._get_VTM_hands()[1] == 1:
                    self._update_status_VTM_1234("place", 1)
                    self.reward = 0
                else:
                    self.reward = PENALTY
            elif action == 3:
                if self._check_facing_pick() and self._get_VTM_hands()[0] == 0:
                    self._update_status_VTM_1234("pick", 0)
                    self.reward = 0
                else:
                    self.reward = PENALTY
            elif action == 4:
                if self._check_facing_place() and self._get_VTM_hands()[0] == 1:
                    self._update_status_VTM_1234("place", 0)
                    self.reward = 0
                else:
                    self.reward = PENALTY
            elif 5 <= action <= 12:
                # action - 5 是VTM要面对的PM的下标
                self.reward = -Config.get_value(PMindex2name[self._get_VTM_facing()] + "." + "distance")[action - 5]
                self._set_VTM_facing(action - 5)
            else:
                print("Error: invalid action")

        elif agent_index == 1:
            if action == 0:
                remain_times = self._get_remain_times()
                remain_times_normal = remain_times[remain_times != -1]
                if len(remain_times_normal) == 0:
                    self.reward = PENALTY
                    return
                min_remain_time = np.min(remain_times_normal)
                delta_time = random.randint(min_remain_time // 2, min_remain_time)
                self._update_remain_times(delta_time)
                self._update_status_0()
                self.reward = -delta_time

            elif action == 1:
                Wafer_indexs = np.where(self.state[0:6] == -1)[0]
                if len(Wafer_indexs) == 0:
                    self.reward = PENALTY
                    return
                Wafer_index = Wafer_indexs[0]
                self._set_Wafer_location(Wafer_index, 6)
                self._set_PM_status(6, 1)
                self._set_PM_remain(6, Config.get_value(PMindex2name[6] + "." + "duration")[1])
                self.reward = 0

            elif action == 2:
                if self._get_buffer_count() > 4:
                    self.reward = PENALTY
                    return
                Wafer_index = self._get_Wafer_index(6)
                self._set_Wafer_location(Wafer_index, 7)
                self._set_PM_status(6, 0)
                self._set_PM_remain(6, -1)
                self.reward = 0

            elif action == 3:
                Wafer_index = self._get_Wafer_index(6)
                if Wafer_index == -1 or self._get_Wafer_status(Wafer_index) != 6:
                    self.reward = PENALTY
                    return
                # 做完了
                self._set_Wafer_location(Wafer_index, -1)
                self._set_Wafer_status(Wafer_index, 0)

                self.reward = AWARD
                self.jobs -= 1
                if self.jobs == 0:
                    self.done = True

        else:
            print("Error: invalid agent index")

    def last(self):
        return self.state, self.reward, self.done, self.info

    def agent_iter(self):
        return ['ATM', 'VTM']