import numpy as np
import pyglet
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import time


class Intestine_path(object):
    viewer = None

    def __init__(self):
        self.windows_size = {'w': 400, 'h': 400}    # 視窗大小
        self.intestine_size = 40                    # 大腸大小
        self.intestine_number = 16                  # 大腸總個數
        self.arm_size = 40                          # 手臂大小
        self.object_size = 20                       # 小磁鐵大小
        self.goal = {'x': 200, 'y': 530, 'l': 80}   # 目的地
        self.first_path = False                     # 要不要隨機地圖設定
        self.distance_have_go = 0                   # 總共加起來的距離
        self.action_step = 0                        # 一回合的總步數
        self.step_distance = 5                      # action 走一步的距離
        self.action_range = self.step_distance * 4  # 手臂能移動之範圍，以小磁鐵為中心
        self.image_table = np.zeros((self.windows_size['w'], self.windows_size['h']))  # 大腸記憶矩陣（拿來訓練用）
        self.maze_table = np.zeros((self.windows_size['w'], self.windows_size['h']))  # 地圖記憶矩陣

        # 視窗方格點個數
        self.how_many_point_inTable = {'x': int((self.windows_size['w'] - self.intestine_size) / self.intestine_size) + 1,
                                       'y': int((self.windows_size['h'] - self.intestine_size) / self.intestine_size) + 1}

        self.training_image_table = np.ones((int(self.windows_size['w'] / self.step_distance),
                                             int(self.windows_size['h'] / self.step_distance),
                                             3), dtype=int) * 255
        self.training_image_table_wall = np.zeros((int(self.windows_size['w'] / self.step_distance),
                                                   int(self.windows_size['h'] / self.step_distance)))

        self.training_image_table_arrived = np.zeros((int(self.windows_size['w'] / self.step_distance),
                                                   int(self.windows_size['h'] / self.step_distance)))
        # print('self.training_image_table', self.training_image_table.shape)

        # 定義座標在視窗的真實位置 --> 真實位置（ex: 400, 400）, 座標位置（ex: 1, 3）
        # 以下均給定座標位置，直到最後印出來時（Viewer內）在更新真實位置
        self.Wtable = np.zeros((self.how_many_point_inTable['x'], self.how_many_point_inTable['y'], 2))     # ex:10,10,2
        for i in range(self.how_many_point_inTable['x']):
            for j in range(self.how_many_point_inTable['y']):
                self.Wtable[i, j, 0] = i * self.intestine_size + self.intestine_size / 2
                self.Wtable[i, j, 1] = j * self.intestine_size + self.intestine_size / 2

        # 隨機初始化起點位置（於座標上）
        self.x_start = np.random.random_integers(0, self.how_many_point_inTable['x'] - 1)
        self.y_start = np.random.random_integers(0, self.how_many_point_inTable['y'] - 1)
        self.start_point = {'x': self.x_start, 'y': self.y_start}

        # 各大腸座標（於座標上）
        self.intestine = []
        for _ in range(self.intestine_number):
            self.intestine.append({'x': self.start_point['x'], 'y': self.start_point['y']})

        # 初始化手臂與膠囊位置（於真實上）
        self.arm = {'x': self.Wtable[self.start_point['x'], self.start_point['y'], 0],
                    'y': self.Wtable[self.start_point['x'], self.start_point['y'], 1],
                    'l': self.arm_size}
        self.object = {'x': self.Wtable[self.start_point['x'], self.start_point['y'], 0],
                       'y': self.Wtable[self.start_point['x'], self.start_point['y'], 1],
                       'l': self.object_size}

    def step(self, action):
        done = False
        r = 0
        # 上
        if action == 0 and self.arm['y'] < self.object['y'] + self.action_range:
            self.arm['y'] += self.step_distance
        # 右上
        elif action == 1 and self.arm['y'] < self.object['y'] + self.action_range \
                and self.arm['x'] < self.object['x'] + self.action_range:
            self.arm['y'] += self.step_distance
            self.arm['x'] += self.step_distance
        # 右
        elif action == 2 and self.arm['x'] < self.object['x'] + self.action_range:
            self.arm['x'] += self.step_distance
        # 右下
        elif action == 3 and self.arm['x'] < self.object['x'] + self.action_range \
                and self.arm['y'] > self.object['y'] - self.action_range:
            self.arm['x'] += self.step_distance
            self.arm['y'] -= self.step_distance
        # 下
        elif action == 4 and self.arm['y'] > self.object['y'] - self.action_range:
            self.arm['y'] -= self.step_distance
        # 左下
        elif action == 5 and self.arm['y'] > self.object['y'] - self.action_range \
                and self.arm['x'] > 0:
            self.arm['x'] -= self.step_distance
            self.arm['y'] -= self.step_distance
        # 左
        elif action == 6 and self.arm['x'] > self.object['x'] - self.action_range:
            self.arm['x'] -= self.step_distance
        # 左上
        elif action == 7 and self.arm['x'] > self.object['x'] - self.action_range \
                and self.arm['y'] < self.object['y'] + self.action_range:       # right
            self.arm['x'] -= self.step_distance
            self.arm['y'] += self.step_distance

        arm_Wtable_axis = {'x': int(self.arm['x'] / self.step_distance),
                           'y': int(self.arm['y'] / self.step_distance)}

        change = False
        obj_InTheIntestine = self.in_range(action)
        object_Wtable_axis = {'x': int(self.object['x'] / self.step_distance),
                              'y': int(self.object['y'] / self.step_distance)}
        if self.arm['x'] - self.arm['l'] / 4 <= self.object['x'] <= self.arm['x'] + self.arm['l'] / 4 and \
                self.arm['y'] - self.arm['l'] / 4 <= self.object['y'] <= self.arm['y'] + self.arm['l'] / 4:
            if obj_InTheIntestine:
                change = True
                self.object['x'] = self.arm['x']
                self.object['y'] = self.arm['y']
                object_Wtable_axis['x'] = int(self.object['x'] / self.step_distance)
                object_Wtable_axis['y'] = int(self.object['y'] / self.step_distance)
                if self.training_image_table_arrived[object_Wtable_axis['x'], object_Wtable_axis['y']] == 0:
                    self.training_image_table_arrived[object_Wtable_axis['x'], object_Wtable_axis['y']] = 1
                    r = 10
                else:
                    r = -10

                if self.training_image_table_wall[object_Wtable_axis['x'], object_Wtable_axis['y']] == 0:
                    self.training_image_table[object_Wtable_axis['x'], object_Wtable_axis['y'], 0] = 0      # R
                    self.training_image_table[object_Wtable_axis['x'], object_Wtable_axis['y'], 1] = 255    # G
                    self.training_image_table[object_Wtable_axis['x'], object_Wtable_axis['y'], 2] = 0      # B

            else:
                self.training_image_table[object_Wtable_axis['x'], object_Wtable_axis['y'], 0] = 255    # R
                self.training_image_table[object_Wtable_axis['x'], object_Wtable_axis['y'], 1] = 0      # G
                self.training_image_table[object_Wtable_axis['x'], object_Wtable_axis['y'], 2] = 0      # B
                self.training_image_table_wall[object_Wtable_axis['x'], object_Wtable_axis['y']] = 1

        pix_size = 2
        state = self.training_image_table.copy()
        while state.shape != self.training_image_table.shape:
            print('fail!', state.shape)
            state = self.training_image_table.copy()
        try:
            state[arm_Wtable_axis['x'] - pix_size:arm_Wtable_axis['x'] + pix_size, arm_Wtable_axis['y'] - pix_size:arm_Wtable_axis['y'] + pix_size, :] = np.zeros((pix_size * 2, pix_size * 2, 3))
            state[object_Wtable_axis['x'] - pix_size:object_Wtable_axis['x'] + pix_size, object_Wtable_axis['y'] - pix_size:object_Wtable_axis['y'] + pix_size, :] = np.zeros((pix_size * 2, pix_size * 2, 3))
            state[arm_Wtable_axis['x'] - pix_size:arm_Wtable_axis['x'] + pix_size, arm_Wtable_axis['y'] - pix_size:arm_Wtable_axis['y'] + pix_size, 2] = np.ones((pix_size * 2, pix_size * 2)) * 255
            state[object_Wtable_axis['x'] - pix_size:object_Wtable_axis['x'] + pix_size, object_Wtable_axis['y'] - pix_size:object_Wtable_axis['y'] + pix_size, 0] = np.ones((pix_size * 2, pix_size * 2)) * 125
            state[object_Wtable_axis['x'] - pix_size:object_Wtable_axis['x'] + pix_size, object_Wtable_axis['y'] - pix_size:object_Wtable_axis['y'] + pix_size, 1] = np.ones((pix_size * 2, pix_size * 2)) * 125
            # state = np.flip(state, 1)
            s = np.rot90(state)

            if change == False:
                r = -((self.arm['x'] - self.object['x']) ** 2 + (self.arm['y'] - self.object['y']) ** 2) ** 0.5
            else:
                r += self.distance_from_start()

            x_goal = self.Wtable[self.goal['x'], self.goal['y'], 0]
            y_goal = self.Wtable[self.goal['x'], self.goal['y'], 1]

            if x_goal - self.intestine_size / 2 <= self.object['x'] <= x_goal + self.intestine_size / 2:
                if y_goal - self.intestine_size / 2 <= self.object['y'] <= y_goal + self.intestine_size / 2:
                    r = 100
                    done = True

            dis_arm_obj = ((self.arm['x'] - self.object['x']) ** 2 + (self.arm['y'] - self.object['y']) ** 2) ** 0.5
            # attraction = 1000 * (1 - math.exp(-1 * dis_arm_obj / 5))

            self.action_step += 1
            # s = self.image_table.flatten()
            # if change:
            #     print('reward: ', r, 'change')
            # else:
            #     print('reward: ', r)
        except:
            print('Env die!')
            s = self.reset(random=True)
            done = True

        s = np.array(s) / 255
        return s, r, done

    def in_range(self, action):
        # 判斷小磁鐵是否在腸道內，並允許小磁鐵被作動
        obj_InTheIntestine = False
        for i in range(self.intestine_number):
            x_point = self.Wtable[self.intestine[i]['x'], self.intestine[i]['y'], 0]
            y_point = self.Wtable[self.intestine[i]['x'], self.intestine[i]['y'], 1]

            if x_point - self.intestine_size / 2 <= self.arm['x'] <= x_point + self.intestine_size / 2:
                if y_point - self.intestine_size / 2 <= self.arm['y'] <= y_point + self.intestine_size / 2:
                    obj_InTheIntestine = True

        return obj_InTheIntestine


    def reset(self, random):
        # 歸零相關數據
        self.distance_have_go = 0
        self.action_step = 0
        self.training_image_table = np.ones((int(self.windows_size['w'] / self.step_distance),
                                             int(self.windows_size['h'] / self.step_distance),
                                             3), dtype=int) * 255
        # 重新安排大腸地圖（於座標上）
        if random:
            # 選擇每次 reset 大腸都不同
            while True:
                if self.reset_intestine(new=True):
                    break
        elif not random and not self.first_path:
            # 選擇每次 reset 大腸都是第一個大腸地圖
            while True:
                if self.reset_intestine(new=True):
                    self.first_path = True
                    break
        elif self.first_path:
            self.reset_intestine(new=False)

        s = self.training_image_table / 255
        return s     # return state

    def reset_intestine(self, new):
        if not new:
            # 初始化手臂與膠囊位置（於真實上）
            self.arm = {'x': self.Wtable[self.start_point['x'], self.start_point['y'], 0],
                        'y': self.Wtable[self.start_point['x'], self.start_point['y'], 1],
                        'l': self.arm_size}
            self.object = {'x': self.Wtable[self.start_point['x'], self.start_point['y'], 0],
                           'y': self.Wtable[self.start_point['x'], self.start_point['y'], 1],
                           'l': self.object_size}

            self.image_table[int(self.arm['x']), int(self.arm['y'])] += 1
            self.image_table[int(self.object['x']), int(self.object['x'])] += 2

        else:

            # 隨機初始化起點位置（於座標上）
            self.x_start = np.random.random_integers(0, self.how_many_point_inTable['x'] - 1)
            self.y_start = np.random.random_integers(0, self.how_many_point_inTable['y'] - 1)
            self.start_point = {'x': self.x_start, 'y': self.y_start}

            # 初始化手臂與膠囊位置（於真實上）
            self.arm = {'x': self.Wtable[self.start_point['x'], self.start_point['y'], 0],
                        'y': self.Wtable[self.start_point['x'], self.start_point['y'], 1],
                        'l': self.arm_size}
            self.object = {'x': self.Wtable[self.start_point['x'], self.start_point['y'], 0],
                           'y': self.Wtable[self.start_point['x'], self.start_point['y'], 1],
                           'l': self.object_size}

            self.image_table[int(self.arm['x']), int(self.arm['y'])] += 1
            self.image_table[int(self.object['x']), int(self.object['x'])] += 2

            # 重新安排大腸地圖
            self.intestine = []
            for _ in range(self.intestine_number):
                self.intestine.append({'x': self.start_point['x'], 'y': self.start_point['y']})

            last_id = 0
            intent_to = 0
            self.Wtable_on = np.zeros((self.how_many_point_inTable['x'], self.how_many_point_inTable['y']))
            self.Wtable_on_for_in_range = np.zeros((self.how_many_point_inTable['x'] + 1, self.how_many_point_inTable['y'] + 1))
            for i in range(self.intestine_number):
                if i % 4 == 3:
                    if self.intestine[i - 1]['x'] < self.how_many_point_inTable['x'] / 2 and self.intestine[i - 1]['y'] < self.how_many_point_inTable['y'] / 2:
                        intent_to = 0  # 於左下角
                    elif self.intestine[i - 1]['x'] < self.how_many_point_inTable['x'] / 2 and self.intestine[i - 1]['y'] > self.how_many_point_inTable['y'] / 2:
                        intent_to = 1  # 於左上角
                    elif self.intestine[i - 1]['x'] > self.how_many_point_inTable['x'] / 2 and self.intestine[i - 1]['y'] > self.how_many_point_inTable['y'] / 2:
                        intent_to = 2  # 於右上角
                    elif self.intestine[i - 1]['x'] > self.how_many_point_inTable['x'] / 2 and self.intestine[i - 1]['y'] < self.how_many_point_inTable['y'] / 2:
                        intent_to = 3  # 於右下角

                if i == 0:
                    self.intestine[0]['x'] = self.start_point['x']
                    self.intestine[0]['y'] = self.start_point['y']
                    self.Wtable_on[self.start_point['x'], self.start_point['y']] = 1
                    self.Wtable_on_for_in_range[self.start_point['x'], self.start_point['y']] = 1
                    if self.start_point['x'] < self.how_many_point_inTable['x'] / 2 and self.start_point['y'] < self.how_many_point_inTable['y'] / 2:
                        intent_to = 0  # 於左下角
                    elif self.start_point['x'] < self.how_many_point_inTable['x'] / 2 and self.start_point['y'] > self.how_many_point_inTable['y'] / 2:
                        intent_to = 1  # 於左上角
                    elif self.start_point['x'] > self.how_many_point_inTable['x'] / 2 and self.start_point['y'] > self.how_many_point_inTable['y'] / 2:
                        intent_to = 2  # 於右上角
                    elif self.start_point['x'] > self.how_many_point_inTable['x'] / 2 and self.start_point['y'] < self.how_many_point_inTable['y'] / 2:
                        intent_to = 3  # 於右下角

                else:
                    # 隨機選擇長出方塊的方向，0：上， 1：下， 2：左， 3：右
                    for j in range(100):
                        if intent_to == 0:      # 期許往右上
                            choose = np.random.choice(4, p=[0.35, 0.15, 0.15, 0.35])
                        elif intent_to == 1:    # 期許往右下
                            choose = np.random.choice(4, p=[0.15, 0.35, 0.15, 0.35])
                        elif intent_to == 2:    # 期許往左下
                            choose = np.random.choice(4, p=[0.15, 0.35, 0.35, 0.15])
                        elif intent_to == 3:    # 期許往左上
                            choose = np.random.choice(4, p=[0.35, 0.15, 0.35, 0.15])

                        situation = {'x': self.intestine[i - 1]['x'], 'y': self.intestine[i - 1]['y']}
                        if choose == 0 and self.intestine[i - 1]['y'] + 1 < self.how_many_point_inTable['y']:
                            situation['y'] = self.intestine[i - 1]['y'] + 1
                        elif choose == 1 and 0 <= self.intestine[i - 1]['y'] - 1:
                            situation['y'] = self.intestine[i - 1]['y'] - 1
                        elif choose == 2 and 0 <= self.intestine[i - 1]['x'] - 1:
                            situation['x'] = self.intestine[i - 1]['x'] - 1
                        elif choose == 3 and self.intestine[i - 1]['x'] + 1 < self.how_many_point_inTable['x']:
                            situation['x'] = self.intestine[i - 1]['x'] + 1

                        if self.Wtable_on[situation['x'], situation['y']] == 0:
                            self.intestine[i]['x'] = situation['x']
                            self.intestine[i]['y'] = situation['y']
                            self.Wtable_on[situation['x'], situation['y']] = 1
                            self.Wtable_on_for_in_range[situation['x'], situation['y']] = 1

                            try:
                                if not (0 < self.intestine[i]['x'] < self.how_many_point_inTable['x'] \
                                        and 0 < self.intestine[i]['y'] < self.how_many_point_inTable['y']):
                                    return False

                                if i == 1:  # 當第二個大確定後，第一個大腸四周關閉
                                    self.Wtable_on[self.intestine[0]['x'] - 1, self.intestine[0]['y']] = 1
                                    self.Wtable_on[self.intestine[0]['x'] + 1, self.intestine[0]['y']] = 1
                                    self.Wtable_on[self.intestine[0]['x'], self.intestine[0]['y'] - 1] = 1
                                    self.Wtable_on[self.intestine[0]['x'], self.intestine[0]['y'] + 1] = 1

                                if choose == 0:      # 如果長腸子動作採取像上長，則上一個腸子的位置左右不允許長腸子(為了合理性)
                                    # 如果腸子往上長，但是上個腸子的位置是一個轉彎處，擇把上一個腸子的下方關閉
                                    if self.Wtable_on[self.intestine[i - 1]['x'] - 1, self.intestine[i - 1]['y']] == 1 \
                                            or self.Wtable_on[self.intestine[i - 1]['x'] + 1, self.intestine[i - 1]['y']] == 1:
                                        self.Wtable_on[self.intestine[i - 1]['x'], self.intestine[i - 1]['y'] - 1] = 1
                                    self.Wtable_on[self.intestine[i - 1]['x'] - 1, self.intestine[i - 1]['y']] = 1
                                    self.Wtable_on[self.intestine[i - 1]['x'] + 1, self.intestine[i - 1]['y']] = 1
                                elif choose == 1:
                                    if self.Wtable_on[self.intestine[i - 1]['x'] - 1, self.intestine[i - 1]['y']] == 1 \
                                            or self.Wtable_on[self.intestine[i - 1]['x'] + 1, self.intestine[i - 1]['y']] == 1:
                                        self.Wtable_on[self.intestine[i - 1]['x'], self.intestine[i - 1]['y'] + 1] = 1
                                    self.Wtable_on[self.intestine[i - 1]['x'] - 1, self.intestine[i - 1]['y']] = 1
                                    self.Wtable_on[self.intestine[i - 1]['x'] + 1, self.intestine[i - 1]['y']] = 1
                                elif choose == 2:
                                    if self.Wtable_on[self.intestine[i - 1]['x'], self.intestine[i - 1]['y'] - 1] == 1 \
                                            or self.Wtable_on[self.intestine[i - 1]['x'], self.intestine[i - 1]['y'] + 1] == 1:
                                        self.Wtable_on[self.intestine[i - 1]['x'] + 1, self.intestine[i - 1]['y']] = 1
                                    self.Wtable_on[self.intestine[i - 1]['x'], self.intestine[i - 1]['y'] - 1] = 1
                                    self.Wtable_on[self.intestine[i - 1]['x'], self.intestine[i - 1]['y'] + 1] = 1
                                elif choose == 3:
                                    if self.Wtable_on[self.intestine[i - 1]['x'], self.intestine[i - 1]['y'] - 1] == 1 \
                                            or self.Wtable_on[self.intestine[i - 1]['x'], self.intestine[i - 1]['y'] + 1] == 1:
                                        self.Wtable_on[self.intestine[i - 1]['x'] - 1, self.intestine[i - 1]['y']] = 1
                                    self.Wtable_on[self.intestine[i - 1]['x'], self.intestine[i - 1]['y'] - 1] = 1
                                    self.Wtable_on[self.intestine[i - 1]['x'], self.intestine[i - 1]['y'] + 1] = 1
                                break
                            except:
                                return False
                        if j == 99:     # 建立模型失敗
                            print('Make model False! Again!')
                            return False
                last_id = i
            # 把最後一段當作終點
            self.goal = self.intestine[last_id]
            return True

    def distance_from_start(self):
        # 判斷小磁鐵是否在腸道內，並允許小磁鐵被作動
        obj_InWhichIntestine = 0
        for i in range(self.intestine_number):
            x_point = self.Wtable[self.intestine[i]['x'], self.intestine[i]['y'], 0]
            y_point = self.Wtable[self.intestine[i]['x'], self.intestine[i]['y'], 1]

            if x_point - self.intestine_size / 2 <= self.object['x'] <= x_point + self.intestine_size / 2:
                if y_point - self.intestine_size / 2 <= self.object['y'] <= y_point + self.intestine_size / 2:
                    obj_InWhichIntestine = i

        return obj_InWhichIntestine


    def render(self, intestine_update):
        if self.viewer is None:
            # 視窗大小、視窗真實位置與座標對應表、機械手臂位置、小磁鐵位置、最後目標位置、大腸數量、大腸新擺設
            self.viewer = Viewer_main(self.windows_size, self.Wtable, self.arm, self.object, self.goal,
                                 self.intestine_number, self.intestine, self.intestine_size)
        self.viewer.render(intestine_update, self.arm, self.object, self.intestine, self.goal)


class Viewer_main(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, windows, Wtable, arm, object, goal, intestine_number, intestine, intestine_size):
        self.windows = windows
        self.Wtable = Wtable
        self.arm = arm
        self.object = object
        self.goal = goal
        self.intestine = intestine
        self.intestine_size = intestine_size
        self.intestine_number = intestine_number
        self.batch = pyglet.graphics.Batch()

        super(Viewer_main, self).__init__(width=self.windows['w'], height=self.windows['h'], resizable=False, caption='Intestine1', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)

        self.intestine_print = []
        for i in range(self.intestine_number):
            self.intestine_print.append(self.batch.add(
                4, pyglet.gl.GL_QUADS, None,
                ('v2f', [160+i, 160+i,
                         160+i, 240+i,
                         240+i, 240+i,
                         240+i, 160+i]),
                ('c3B', (250, 85, 85) * 4)))

        self.goal_print = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.goal['x'] - self.intestine_size / 2, self.goal['y'] - self.intestine_size / 2,
                     self.goal['x'] - self.intestine_size / 2, self.goal['y'] + self.intestine_size / 2,
                     self.goal['x'] + self.intestine_size / 2, self.goal['y'] + self.intestine_size / 2,
                     self.goal['x'] + self.intestine_size / 2, self.goal['y'] - self.intestine_size / 2]),
            ('c3B', (80, 80, 240) * 4))

        self.arm_print = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.arm['x'] - self.arm['l'] / 2, self.arm['y'] - self.arm['l'] / 2,
                     self.arm['x'] - self.arm['l'] / 2, self.arm['y'] + self.arm['l'] / 2,
                     self.arm['x'] + self.arm['l'] / 2, self.arm['y'] + self.arm['l'] / 2,
                     self.arm['x'] + self.arm['l'] / 2, self.arm['y'] - self.arm['l'] / 2]),
            ('c3B', (80, 240, 80) * 4))

        self.object_print = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [self.object['x'] - self.object['l'] / 2, self.object['y'] - self.object['l'] / 2,
                     self.object['x'] - self.object['l'] / 2, self.object['y'] + self.object['l'] / 2,
                     self.object['x'] + self.object['l'] / 2, self.object['y'] + self.object['l'] / 2,
                     self.object['x'] + self.object['l'] / 2, self.object['y'] - self.object['l'] / 2]),
            ('c3B', (0, 255, 255) * 4))

    def render(self, intestine_update, arm, object, intestine, goal):
        self._update_arm(arm, object)
        if intestine_update:
            self._update_intestine(intestine, goal)
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self, arm, object):
        self.arm = arm
        self.object = object
        xy11_arm = [self.arm['x'] - self.arm['l'] / 2, self.arm['y'] - self.arm['l'] / 2]
        xy12_arm = [self.arm['x'] - self.arm['l'] / 2, self.arm['y'] + self.arm['l'] / 2]
        xy21_arm = [self.arm['x'] + self.arm['l'] / 2, self.arm['y'] + self.arm['l'] / 2]
        xy22_arm = [self.arm['x'] + self.arm['l'] / 2, self.arm['y'] - self.arm['l'] / 2]
        self.arm_print.vertices = np.concatenate((xy11_arm, xy12_arm, xy21_arm, xy22_arm))

        xy11_obj = [self.object['x'] - self.object['l'] / 2, self.object['y'] - self.object['l'] / 2]
        xy12_obj = [self.object['x'] - self.object['l'] / 2, self.object['y'] + self.object['l'] / 2]
        xy21_obj = [self.object['x'] + self.object['l'] / 2, self.object['y'] + self.object['l'] / 2]
        xy22_obj = [self.object['x'] + self.object['l'] / 2, self.object['y'] - self.object['l'] / 2]
        self.object_print.vertices = np.concatenate((xy11_obj, xy12_obj, xy21_obj, xy22_obj))

    def _update_intestine(self, intestine, goal):
        self.intestine = intestine
        self.goal = goal
        for i in range(self.intestine_number):
            x = self.Wtable[self.intestine[i]['x'], self.intestine[i]['y'], 0]
            y = self.Wtable[self.intestine[i]['x'], self.intestine[i]['y'], 1]
            xy11_intestine = [x - self.intestine_size / 2, y - self.intestine_size / 2]
            xy12_intestine = [x - self.intestine_size / 2, y + self.intestine_size / 2]
            xy21_intestine = [x + self.intestine_size / 2, y + self.intestine_size / 2]
            xy22_intestine = [x + self.intestine_size / 2, y - self.intestine_size / 2]
            self.intestine_print[i].vertices = np.concatenate((xy11_intestine, xy12_intestine, xy21_intestine, xy22_intestine))

        x = self.Wtable[self.goal['x'], self.goal['y'], 0]
        y = self.Wtable[self.goal['x'], self.goal['y'], 1]
        xy11_goal = [x - self.intestine_size / 2, y - self.intestine_size / 2]
        xy12_goal = [x - self.intestine_size / 2, y + self.intestine_size / 2]
        xy21_goal = [x + self.intestine_size / 2, y + self.intestine_size / 2]
        xy22_goal = [x + self.intestine_size / 2, y - self.intestine_size / 2]
        self.goal_print.vertices = np.concatenate((xy11_goal, xy12_goal, xy21_goal, xy22_goal))


if __name__ == '__main__':
    show = True
    env = Intestine_path()

    if show:
        plt.ion()
        plt.figure()

    while True:
        env.reset(random=True)
        env.render(True)

        # env.viewer.set_vsync(True)
        for i in range(10000):
            env.render(False)
            s, r, done = env.step(np.random.random_integers(0, 7))
            s = torch.from_numpy(np.array(s))
            if i % 2000 == 0 and show:
                plt.imshow(s)



            # env.step(1)
            # input('pass')
            # if i < 25:
            #     env.step(2)
            # elif i < 75:
            #     env.step(3)
            # elif i < 100:
            #     env.step(0)
            # elif i < 150:
            #     env.step(1)