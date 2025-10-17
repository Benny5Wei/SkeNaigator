import cv2
import numpy as np
import skfmm
from numpy import ma
import copy

def get_mask(sx, sy, scale, step_size):
    """
    using all the points on the edges as mask.
    """
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size))
    """
    mask[0, size//2] = 1
    mask[-1, size//2] = 1
    mask[size//2, 0] = 1
    mask[size//2, -1] = 1

    mask[0, size//4] = 1
    mask[-1, size//4] = 1
    mask[size//4, 0] = 1
    mask[size//4, -1] = 1

    mask[0, 3*size//4] = 1
    mask[-1, 3*size//4] = 1
    mask[3*size//4, 0] = 1
    mask[3*size//4, -1] = 1

    mask[0, 0] = 1
    mask[-1, -1] = 1
    mask[0, -1] = 1
    mask[-1, 0] = 1
    """
    # 将mask的第一行和最后一行全部置为1
    mask[0,:] = mask[-1,:] = 1
    # 将mask的第一列和最后一列全部置为1
    mask[:,0] = mask[:,-1] = 1
    # 将mask的中心点置为1
    mask[size//2, size//2] = 1
    return mask


def get_dist(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    # 创建一个全为1e-10的掩码
    mask = np.zeros((size, size)) + 1e-10
    # 遍历掩码的每个元素
    for i in range(size):
        for j in range(size):
                # 计算每个元素到中心点的距离
                mask[i, j] = max(5, (((i + 0.5) - (size // 2 + sx)) ** 2 +
                                 ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
    # 返回掩码
    return mask


def moving_avg(a, n=2):
    # 获取输入矩阵的行数和列数
    h = a.shape[0]
    w = a.shape[1]
    # 创建一个新的矩阵b，大小为输入矩阵减去2n
    b = copy.deepcopy(a[n:h-n,n:w-n])
    # 遍历n到2n之间的所有数
    for i in range(1,n+1):
        # 遍历1到i之间的所有数
        for j in range(1,i+1):
            # 将输入矩阵中四个方向的元素加到b中，并除以800*i
            b += a[n+j:h-n+j,n+(i-j):w-n+(i-j)] / (800*i)
            b += a[n-(i-j):h-n-(i-j),n+j:w-n+j]/ (800*i)
            b += a[n-j:h-n-j,n-(i-j):w-n-(i-j)]/ (800*i)
            b += a[n+(i-j):h-n+(i-j),n-j:w-n-j]/ (800*i)
    # 将b中的元素除以1+n/200
    b /= 1+n/200

    return b

class FMMPlanner():
    def __init__(self, traversible, args, scale=1, step_size=5):
        self.scale = scale
        self.args = args
        self.step_size = step_size
        self.visualize = False
        self.stop_cond = 0.5
        self.save = False
        self.save_t = 0
        if scale != 1.:
            self.traversible = cv2.resize(traversible,
                                          (traversible.shape[1] // scale,
                                           traversible.shape[0] // scale),
                                          interpolation=cv2.INTER_NEAREST)
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.du = int(self.step_size / (self.scale * 1.))

    def set_goal(self, goal, auto_improve=False):
        # 将traversible矩阵中的0值进行掩码处理
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        # 将goal的坐标转换为整数
        goal_x, goal_y = int(goal[0] / (self.scale * 1.)), \
                         int(goal[1] / (self.scale * 1.))

        # 如果goal的坐标在traversible矩阵中为0，则找到最近的goal
        if self.traversible[goal_x, goal_y] == 0.:
            goal_x, goal_y = self._find_nearest_goal([goal_x, goal_y])

        # 将goal的坐标在traversible矩阵中置为0
        traversible_ma[goal_x, goal_y] = 0
        # 计算traversible矩阵中每个点到goal的距离
        dd = skfmm.distance(traversible_ma, dx=1) # time consuming
        # 将距离矩阵中的掩码值填充为距离矩阵中的最大值加1
        dd = ma.filled(dd, np.max(dd) + 1)
        # 将距离矩阵赋值给self.fmm_dist
        self.fmm_dist = dd
        return

    def set_multi_goal(self, goal_map, state):
        # 将traversible矩阵中的0值替换为1，1值替换为0
        traversible_ma = ma.masked_values(self.traversible * 1, 0) # travrsible 1 and false
        # 获取goal_map中值为1的坐标
        goal_x, goal_y = np.where(goal_map==1)
        # 如果goal_map中值为1的坐标在traversible矩阵中为0，则找到最近的traversible坐标作为goal
        if self.traversible[goal_x, goal_y] == 0.: 
            ## if goal is not traversible, find a traversible place nearby as goal
            goal_x, goal_y = self._find_nearest_goal([goal_x, goal_y], state)
            
        # 将goal坐标在traversible矩阵中置为0
        traversible_ma[goal_x, goal_y] = 0
        # 计算traversible矩阵中每个点到goal坐标的距离
        dd = skfmm.distance(traversible_ma, dx=1)
        
        # 将traversible矩阵中未遍历到的坐标填充为最大距离
        dd = ma.filled(dd, np.max(dd) + 1) # fill untraversible place as max distance
        # 如果agent在未遍历到的坐标上，则将goal_map中值为1的坐标置为0，并重新计算距离
        if dd[state[0],state[1]] == np.max(dd): # agent is in a untraversible place (not supposed to happen)
            goal_map_ma = np.zeros_like(goal_map) == 0
            goal_map_ma[goal_map == 1] = 0
            dd += skfmm.distance(goal_map_ma, dx=1)
        # 将计算出的距离赋值给fmm_dist
        self.fmm_dist = dd
   
        return

    def get_short_term_goal(self, state, found_goal = 0, decrease_stop_cond=0):
        # 获取短期目标
        scale = self.scale * 1.
        # 将状态中的每个元素除以缩放因子
        state = [x / scale for x in state]
        # 计算状态中的x和y坐标与整数部分的差值
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        # 获取掩码
        mask = get_mask(dx, dy, scale, self.step_size)
        # 获取距离掩码
        dist_mask = get_dist(dx, dy, scale, self.step_size)

        # 将状态中的每个元素转换为整数
        state = [int(x) for x in state]
        n = 2
        # 获取fmm距离
        dist = np.pad(self.fmm_dist, self.du + n,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)
        # 获取子集
        subset = dist[state[0]:state[0] + 2 * self.du + 1,
                      state[1]:state[1] + 2 * self.du + 1]
        # 获取大子集
        subset_large = dist[state[0]:state[0] + 2 * self.du + 1+2*n,
                      state[1]:state[1] + 2 * self.du + 1+2*n]
        # 对大子集进行移动平均
        subset = moving_avg(subset_large, n=n)
        # 断言子集的形状
        assert subset.shape[0] == 2 * self.du + 1 and \
               subset.shape[1] == 2 * self.du + 1, \
            "Planning error: unexpected subset shape {}".format(subset.shape)

        # 将子集乘以掩码
        subset *= mask
        # 将子集加上掩码的补数
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2
        
        # 获取停止条件
        stop_condition = max((self.stop_cond - decrease_stop_cond)*100/5., 0.2)

        # 如果可视化
        if self.visualize:
            print("dist until goal is ", subset[self.du, self.du])
        # 如果子集的中心点小于停止条件
        if subset[self.du, self.du] < stop_condition:
            stop = True
        else:
            stop = False

        # 将子集减去子集的中心点
        subset -= subset[self.du, self.du] # dis change wrt agent point
        #ratio1 = subset / dist_mask
        #subset[ratio1 < -1.5] = 1

        ## see which direction has the fastest distance decrease to the goal
        # 获取子集的中心点
        mid = self.du
        # 对子集的每一行进行归一化
        for i in range(len(subset)-1):
            subset[0,i] /= np.sqrt(np.abs(mid-i)**2+mid**2) / mid
        # 对子集的每一列进行归一化
        for i in range(len(subset)-1):
            subset[i,-1] /= np.sqrt(np.abs(mid-i)**2+mid**2) / mid
        # 对子集的每一行进行归一化
        for i in range(len(subset)-1):
            subset[-1,-i-1] /= np.sqrt(np.abs(mid-i)**2+mid**2) / mid
        # 对子集的每一列进行归一化
        for i in range(len(subset)-1):
            subset[-i-1,0] /= np.sqrt(np.abs(mid-i)**2+mid**2) / mid
        
        # do not move accross the obstacles
        
        # 获取子集中距离最小的点的坐标
        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        # 如果子集中距离最小的点大于-0.0001
        if subset[stg_x, stg_y] > -0.0001:
            replan = True
        else:
            replan = False


        # 返回短期目标的坐标和是否需要重新规划
        return (stg_x + state[0] - self.du) * scale, \
               (stg_y + state[1] - self.du) * scale, replan, stop

    def _find_nearest_goal(self, goal, state=None):
        """
        find the nearest traversible place
        """
        # 获取traversible的形状
        max_x, max_y = self.traversible.shape
        # 计算目标点的左上角和右下角坐标
        top_left_selected = (max(0,goal[0]-80), max(0,goal[1]-80))
        down_right_selected = (min(max_x-1, goal[0]+80), min(max_y-1, goal[1]+80))
        
        # 创建一个与目标点区域大小相同的traversible矩阵
        traversible = np.ones((int(down_right_selected[0]-top_left_selected[0]),int(down_right_selected[1]-top_left_selected[1]))) * 1.0
        # 创建FMMPlanner对象
        planner = FMMPlanner(traversible, self.args)
        # 将目标点坐标转换为相对于目标点区域的坐标
        goal = (goal[0]-top_left_selected[0],goal[1]-top_left_selected[1])
        # 设置目标点
        planner.set_goal(goal)
        # 获取目标点区域的traversible矩阵
        mask = self.traversible[int(top_left_selected[0]):int(down_right_selected[0]), int(top_left_selected[1]):int(down_right_selected[1])]

        # 计算FMM距离矩阵
        dist_map = planner.fmm_dist * mask
        # 将距离矩阵中为0的值设置为距离矩阵的最大值的两倍
        dist_map[dist_map == 0] = dist_map.max() * 2
        # 对距离矩阵进行排序
        dist_sort_idx = np.argsort(dist_map, axis=None) # a little time cosuming 
        i = 0
        # 获取距离矩阵中最小值的坐标
        goal = np.unravel_index(dist_sort_idx[i], dist_map.shape)
        # 将坐标转换为相对于原始traversible矩阵的坐标
        goal = (top_left_selected[0]+goal[0],top_left_selected[1]+goal[1])
        # 返回最近的可行点
        return goal
        