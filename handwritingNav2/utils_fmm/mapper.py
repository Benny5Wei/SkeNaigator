# -*- coding: utf-8 -*-
import torch
import skimage
import copy
import cv2
import numpy as np
from matplotlib import colors

# 修改为包导入形式
from utils_fmm.mapping import Semantic_Mapping
import utils_fmm.control_helper as CH
import utils_fmm.pose_utils as pu
from torchvision.utils import save_image

class Mapper(object):

    def __init__(self, config, device):
        self.config = config
        self.device = device

        ### ----- init some static variables ----- ###
        self.map_size_cm = 5120
        self.map_resolution = 10 # 10
        self.map_size = self.map_size_cm // self.map_resolution

        self.origins = np.zeros((2))
        self.camera_horizon = 0
        self.collision_threshold = 0.08
        self.col_width = 5
        self.selem = skimage.morphology.square(1)
        
        ### ----- init maps ----- ###
        self.reset()
        self.sem_map_module = Semantic_Mapping(config, self.device, self.map_size_cm, self.map_resolution).to(self.device) 
        self.free_map_module = Semantic_Mapping(config, self.device, self.map_size_cm, self.map_resolution, max_height=20,min_height=-150).to(self.device)

        # Adjust height range to match observed point-cloud z around 140‒175 cm
        # Include a small buffer under/over to ensure ground and a bit above it are considered free
        # self.free_map_module = Semantic_Mapping(
        #     config,
        #     self.device,
        #     self.map_size_cm,
        #     self.map_resolution,
        #     max_height=180,   # ~1.8 m
        #     min_height=120    # ~1.2 m (just below typical agent height)
        #).to(self.device)

        self.sem_map_module.eval()
        self.free_map_module.eval()
        self.sem_map_module.set_view_angles(self.camera_horizon)
        self.free_map_module.set_view_angles(self.camera_horizon)

    def reset(self):
        # 重置地图
        self.full_map = torch.zeros(1,1 ,self.map_size, self.map_size).float().to(self.device)

        # 将地图转换为numpy数组
        self.visited = self.full_map[0,0].cpu().numpy()
        self.collision_map = self.full_map[0,0].cpu().numpy()
        # 深度拷贝地图
        self.fbe_free_map = copy.deepcopy(self.full_map).to(self.device) # 0 is unknown, 1 is free
        # 初始化位置
        self.full_pose = torch.zeros(3).float().to(self.device)
        self.last_loc = self.full_pose
        # 初始化轨迹
        self.agent_traj = []
        # 初始化动作
        self.prev_action = 0
        # Origin of local map
        
        # 初始化地图和位置
        def init_map_and_pose(): 
            # 将地图填充为0
            self.full_map.fill_(0.)
            # 将位置填充为0
            self.full_pose.fill_(0.)
            # 将位置的前两个元素设置为地图大小的中点
            self.full_pose[:2] = self.map_size_cm / 100.0 / 2.0  # put the agent in the middle of the map

        init_map_and_pose()
        

    def update_map(self, observations):
        """
        full pose: gps and angle in the initial coordinate system, where 0 is towards the x axis
        """
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0 + observations['gps'].to(self.device)[0, 0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0 - observations['gps'].to(self.device)[0, 1]
        self.full_pose[2:] = (observations['compass']* 57.29577951308232).to(self.device) # input degrees and meters
        self.full_map = self.sem_map_module(torch.squeeze(observations['depth'][0], dim=-1).to(self.device), self.full_pose, self.full_map)
    
    def update_free_map(self, observations):
        """
        update free map using visual projection
        """
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0+observations['gps'].to(self.device)[0, 0]
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0-observations['gps'].to(self.device)[0, 1]
        self.full_pose[2:] = (observations['compass'] * 57.29577951308232).to(self.device) # input degrees and meters
        self.fbe_free_map = self.free_map_module(torch.squeeze(observations['depth'][0], dim=-1).to(self.device), self.full_pose, self.fbe_free_map)
        self.fbe_free_map[int(self.map_size_cm / 10) - 3:int(self.map_size_cm / 10) + 4, int(self.map_size_cm / 10) - 3:int(self.map_size_cm / 10) + 4] = 1
        print("free sum:", torch.sum(self.fbe_free_map).item())
    
    def get_traversible(self, map_pred, pose_pred):
        """
        update traversible map
        """
        grid = np.rint(map_pred)

        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        gx1, gx2, gy1, gy2  = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]

        r, c = start_y, start_x
        start = [int(r*100/self.map_resolution - gy1), int(c*100/self.map_resolution - gx1)]
        start = pu.threshold_poses(start, grid.shape)

        self.visited[gy1:gy2, gx1:gx2][start[0]-2:start[0]+3, start[1]-2:start[1]+3] = 1
        # Get traversible
        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h+2,w+2)) + value
            new_mat[1:h+1,1:w+1] = mat
            return new_mat

        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape
        traversible = skimage.morphology.binary_dilation(grid[y1:y2, x1:x2], self.selem) != True

        traversible = 1 - traversible
        selem = skimage.morphology.disk(4)
        traversible = skimage.morphology.binary_dilation(traversible, selem) != True

        traversible[int(start[0]-y1)-1:int(start[0]-y1)+2, int(start[1]-x1)-1:int(start[1]-x1)+2] = 1
        traversible = traversible * 1.
        traversible[self.visited[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 1
        traversible[self.collision_map[gy1:gy2, gx1:gx2][y1:y2, x1:x2] == 1] = 0
        traversible = add_boundary(traversible)
        return traversible, start, start_o

        
    def forward(self, observations, action_per_env= None, show_obstacle = True, show_visited_area = False, show_frontier = False):
        max_d = 5.0     # 5.0
        depth = observations["depth"]  # H×W, float32

        invalid = (depth >= max_d - 1e-3) | (depth <= 1e-4)   # ≥5 m 或 0
        depth[invalid] = 0.0          # 0 → unknown, 在体素化时会被忽略
        # --- debugging: save current maps to measure delta ---
        # old_full_map = self.full_map.clone()
        # old_free_map = self.fbe_free_map.clone()
        self.prev_action = action_per_env[0]
        self.update_map(observations)
        self.update_free_map(observations)
        # Print how many grid cells changed this step
        # delta_map = torch.sum(torch.abs(self.full_map - old_full_map)).item()
        # delta_free = torch.sum(torch.abs(self.fbe_free_map - old_free_map)).item()
        # print(f"Map delta: {delta_map}, Free map delta: {delta_free}")
        # self.agent_traj.append([int((self.map_size_cm/100-self.full_pose[1])*100/self.map_resolution), int(self.full_pose[0]*100/self.map_resolution)])

        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy()
        input_pose[1] = self.map_size_cm/100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]

        traversible, _, __= self.get_traversible(self.full_map.cpu().numpy()[0,0,::-1], input_pose)
        
        map = copy.deepcopy(torch.from_numpy(traversible))
        gray_map = torch.stack((map, map, map))
        paper_obstacle_map = copy.deepcopy(gray_map)[:,1:-1,1:-1] # C H W
        # save_image(paper_obstacle_map.to(dtype=torch.float32), "/data/xhj/handwritingNav/figures/map.png")
        semantic_map = torch.zeros_like(paper_obstacle_map).permute(1,2,0) # H W C

        unknown_rgb = colors.to_rgb('white')
        unknown_rgb_tensor = torch.tensor(unknown_rgb)
        semantic_map[:,:,:] = unknown_rgb_tensor
        
        # 1) draw obstacles first so later layers (visited/free/frontier) stay visible


        # 2) draw visited / free area
        if show_visited_area:
            free_rgb = colors.to_rgb('lightgreen')
            #free_rgb = colors.to_rgb('white')
            free_map = self.fbe_free_map.cpu().numpy()[0,0,::-1].copy() > 0.5  # H x W numpy bool
            mask = torch.from_numpy(free_map).bool()  # 与 semantic_map 前两维相同
            semantic_map[mask] = torch.tensor(free_rgb, dtype=semantic_map.dtype)
            # Debug: print how many pixels were colored as free space
            print("colored free pixels:", mask.sum().item())

            frontier_rgb = colors.to_rgb('indianred')
            selem = skimage.morphology.disk(1)
            free_map[skimage.morphology.binary_dilation(free_map, selem)] = 1
            semantic_map[(free_map==1)*(semantic_map[:,:,0]==unknown_rgb_tensor[0]).numpy(),:] = torch.tensor(frontier_rgb).double()


        if show_obstacle:
            obstacle_rgb = colors.to_rgb('dimgrey')
            obstacle_rgb_tensor = torch.tensor(obstacle_rgb).double()
            obstacle_mask = skimage.morphology.binary_dilation(self.full_map.cpu().numpy()[0,0,::-1]>0.5,
                                                              skimage.morphology.disk(1))
            semantic_map[obstacle_mask,:] = obstacle_rgb_tensor
        # 3) draw exploration frontier
            
        # 注释掉绘制轨迹的代码
        # if len(self.agent_traj) > 0:
        #     semantic_map_np = np.ascontiguousarray((semantic_map.numpy() * 255).astype(np.uint8))
        #     # red_bgr = (255, 0, 0)
        #     red_bgr = (139, 0, 0)
        #     # purple_bgr = (255, 0, 255)
        #     purple_bgr = (48, 25, 52)

        #     if len(self.agent_traj) > 1:
        #         for i in range(1, len(self.agent_traj)):
        #             p1, p2 = tuple(self.agent_traj[i-1]), tuple(self.agent_traj[i])
        #             cv2.line(semantic_map_np, (p1[1], p1[0]), (p2[1], p2[0]), red_bgr, thickness=1)
        #         end = self.agent_traj[-1]
        #         cv2.circle(semantic_map_np, (end[1], end[0]), radius=4, color=purple_bgr, thickness=-1)

        #     start = self.agent_traj[0]
        #     cv2.circle(semantic_map_np, (start[1], start[0]), radius=4, color=red_bgr, thickness=-1)

        #     semantic_map = torch.from_numpy(semantic_map_np.astype(np.float32) / 255.0)
        
        
        # save_image(semantic_map.permute(2,0,1)/semantic_map.max().to(dtype=torch.float32), "/data/xhj/handwritingNav/figures/MAP_1.png")
        self.last_loc = copy.deepcopy(self.full_pose)
        return semantic_map.to(dtype=torch.float).unsqueeze(0)