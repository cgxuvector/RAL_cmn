from env.habitat_env import House
from model.local_map_predictor import CoarseLocalMapPredictor
import numpy as np
import torch
import torchvision
import random
from torchvision.transforms import Normalize
from skimage.transform import resize
from skimage.color import rgb2gray
from utils.Map import TopoMap
from utils.AnyTree import TreeNode, BFTree
from utils.Params import ParamsLoader
import matplotlib.pyplot as plt
import logging
import habitat_sim
import os
import cv2
from utils.AStar import A_star
import IPython.terminal.debugger as Debug


def load_environment_names(env_set_name, mode, seed):
    # load the environment names
    if env_set_name == "replica":
        with open("roughmaps/replica/env_replica_15_names.txt") as f:
            data = f.readlines()
        f.close()
        names = [d.rstrip() for d in data]
    elif env_set_name == "gibson":
        if mode == "full":
            with open("roughmaps/gibson/env_gibson_45_names.txt") as f:
                data = f.readlines()
            f.close()
        else:
            with open(f"roughmaps/gibson/run_seed_{seed}_{mode}.txt") as f:
                data = f.readlines()
            f.close()
        names = [d.rstrip().split(".")[0] for d in data]
    else:
        raise Exception("Invalid dataset name.")

    return names


def make_env(name, configs):
    # station name
    station_name = configs['run_cfg']['station_name']

    # dataset name
    dataset_name = configs['run_cfg']['dataset_name']

    # make the scene absolute path
    if dataset_name == "replica":
        if station_name == "obsidian":
            configs['scene_cfg']["scene"] = f"/mnt/data/cheng_results/habitat_data/envs/replica/" \
                                            f"{name}/habitat/mesh_semantic.ply"  # obsidian
        elif station_name == "desktop":
            configs['scene_cfg']["scene"] = f"/mnt/sda/habitat_datasets/replica/" \
                                            f"{name}/habitat/mesh_semantic.ply"  # desktop
        elif station_name == "panzer":
            configs['scene_cfg']["scene"] = f"/home/chengguang/rlEnvs/replica_v1/" \
                                            f"{name}/habitat/mesh_semantic.ply"
        else:
            raise Exception("Invalid station name.")
    else:
        if station_name == "obsidian":
            configs['scene_cfg']['scene'] = f"/mnt/data/cheng_results/habitat_data/envs/gibson/" \
                                            f"{name}.glb"  # obsidian
        elif station_name == "desktop":
            configs['scene_cfg']["scene"] = f"/mnt/sda/habitat_datasets/Gibson/gibson_habitat/gibson/" \
                                            f"{name}.glb"  # desktop
        elif station_name == "panzer":
            configs['scene_cfg']["scene"] = f"/home/chengguang/rlEnvs/gibson_v1/" \
                                            f"{name}.glb"  # panzer
        else:
            raise Exception("Invalid station name.")

    # create the environment in Habitat
    house = House(scene=configs['scene_cfg']['scene'],  # scene configuration
                  rnd_seed=configs['scene_cfg']['random_seed'],
                  allow_sliding=configs['scene_cfg']['allow_sliding'],
                  max_episode_length=configs['scene_cfg']['max_episode_length'],
                  goal_reach_eps=configs['scene_cfg']['goal_reach_eps'],
                  # observation configuration
                  enable_rgb=True if "color_sensor" in configs['sensor_cfg']["use_sensors"] else False,
                  enable_depth=True if "depth_sensor" in configs['sensor_cfg']["use_sensors"] else False,
                  enable_rgb_depth=True if "color_depth_sensor" in configs['sensor_cfg']["use_sensors"] else False,
                  enable_semantic=True if "semantic_sensor" in configs['sensor_cfg']["use_sensors"] else False,
                  enable_panorama=configs['sensor_cfg']['enable_panorama'],
                  sensor_height=configs['sensor_cfg']['sensor_height'],
                  depth_clip_vmax=configs['sensor_cfg']['clip_depth_max'],
                  obs_width=configs['sensor_cfg']['obs_width'],
                  obs_height=configs['sensor_cfg']['obs_height'],
                  # map configuration
                  top_down_type=configs['map_cfg']['top_down_type'],
                  map_meters_per_pixel=configs['map_cfg']['meters_per_pixel'],
                  local_map_size=configs['map_cfg']['local_map_size'],
                  enable_local_map=configs['map_cfg']['enable_local_map'],
                  enable_ego_local_map=configs['map_cfg']['enable_ego_local_map'],
                  map_show_agent=configs['map_cfg']['show_agent'],
                  map_show_goal=configs['map_cfg']['show_goal'],
                  # agent configuration
                  move_forward_amount=configs['agent_cfg']['move_forward'],
                  turn_left_amount=configs['agent_cfg']['turn_left'],
                  turn_right_amount=configs['agent_cfg']['turn_right']
                  )
    return house


class CoarseMapNavigator(object):
    def __init__(self, params):
        # save the parameters
        self.params = params

        # set the device
        self.device = torch.device(self.params['run_cfg']['device'])

        # save the environment
        self.env_name = None
        self.env = None

        """ Graph-structured 2-D coarse map """
        # graph-structured 2-D coarse map
        self.rough_map = None

        """ Local map predictor """
        self.local_map_predictor = CoarseLocalMapPredictor(self.params['cmn_cfg'])
        self.local_map_predictor.load_state_dict(torch.load(self.params['cmn_cfg']['local_map_predictor_model'],
                                                            map_location="cpu"))
        self.local_map_predictor = self.local_map_predictor.to(self.device)
        self.local_map_predictor.eval()

        """ Noisy Bayesian filter """
        # noisy Bayesian filter: beliefs
        self.predictive_belief_map = None
        self.observation_prob_map = None
        self.updated_belief_map = None
        self.agent_belief_map = None
        self.last_agent_belief_map = None
        self.wall_loc_map = None
        self.goal_loc_rough_map = None
        self.noise_trans_prob = None

        """ Heuristic planner """
        self.breadth_first_tree = None

        # extra variables
        self.current_local_map = np.ones((32, 32))

        """ For visualization """
        self.artists = []
        self.vis_mode = params['run_cfg']['visualization']
        if params['run_cfg']['visualization'] == "policy_and_belief":
            self.vis_fig, self.sub_plots_arr = plt.subplots(1, 2, figsize=(12, 8))
            self.vis_fig.suptitle("Visualize Real Time Policy v.s. Belief")
        elif params['run_cfg']['visualization'] == "belief_update":
            self.vis_fig, self.sub_plots_arr = plt.subplots(2, 3, figsize=(12, 8))
            self.vis_fig.suptitle("Belief update")
        else:
            pass

        """ For transformation """
        self.trans_resize = torchvision.transforms.Resize((224, 224))
        self.trans_normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def load_hand_drawn_map(self):
        # Read the hand-drawing map
        img = cv2.imread(f"roughmaps/gibson_drawn/{self.env_name}.jpeg")

        # Find all the boundary
        lower_bound = np.array([130, 30, 130])
        upper_bound = np.array([250, 250, 250])
        bound_mask = cv2.inRange(img, lower_bound, upper_bound)  # obtain the mask
        bound_img = cv2.bitwise_and(img, img, mask=bound_mask)

        # Convert the image to gray
        map_gray_img = cv2.cvtColor(bound_img, cv2.COLOR_BGR2GRAY)

        # Fill the holes using thresholding
        _, im_th = cv2.threshold(map_gray_img, 0, 255, cv2.THRESH_BINARY)

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_th, mask, (0, 0), 255)

        return im_th

    def init_graph_coarse_map(self):
        # save the environment boundary
        self.params['map_cfg']['env_boundary'] = self.env.simulator.pathfinder.get_bounds()

        # load the 2-D gt map
        gt_map = np.load(f"roughmaps/{self.params['run_cfg']['dataset_name']}/"
                         f"{self.env_name}_map_binary_arr_mpp_015.npy")

        # load the 2-D coarse map
        coarse_map = self.load_hand_drawn_map()

        # Resize the coarse map to the GT map size
        coarse_map_resized = resize(coarse_map, gt_map.shape)
        coarse_map_resized = np.where(coarse_map_resized >= 0.8, 1.0, 0.0)

        # create the graph-structured map
        self.rough_map = TopoMap(coarse_map_resized, self.params['map_cfg'])

    def init_belief(self):
        # init the belief: 1.0 for empty space and 0.0 for walls
        init_space_belief = self.rough_map.map_sampled_grid.copy()

        # compute and set initial probability value
        init_val = 1.0 / len(self.rough_map.local_maps)

        # initialize the belief maps
        self.agent_belief_map = np.where(init_space_belief == 1.0, init_val, 0.0)
        self.last_agent_belief_map = self.agent_belief_map.copy()

        # record the space map
        self.wall_loc_map = self.rough_map.map_sampled_grid.copy()

        # goal location
        self.goal_loc_rough_map = self.render_local_map(self.env.goal_loc)[1]

        # init other beliefs
        rough_map_shape = self.rough_map.map_sampled_grid.shape
        self.predictive_belief_map = np.zeros(rough_map_shape)  # predictive probability
        self.observation_prob_map = np.zeros(rough_map_shape)  # observation probability
        self.updated_belief_map = np.zeros(rough_map_shape)  # updated probability

    @staticmethod
    def compute_l2_dist(loc_1, loc_2):
        return np.linalg.norm(np.array(loc_1) - np.array(loc_2))

    @staticmethod
    def local_map_rgb2binary(loc_m):
        # convert to grayscale
        gray_img = rgb2gray(loc_m)
        # threshold the image
        loc_m = np.where(gray_img < 1.0, 0.0, gray_img)
        return resize(loc_m, (32, 32))

    @staticmethod
    def compute_norm_heuristic_vec(loc_1, loc_2):
        arr_1 = np.array(loc_1)
        arr_2 = np.array(loc_2)
        heu_vec = arr_2 - arr_1
        return heu_vec / np.linalg.norm(heu_vec)

    @staticmethod
    def normalize_belief_for_visualization(belief):
        v_min = belief.min()
        v_max = belief.max()
        belief = (belief - v_min) / (v_max - v_min)
        return np.clip(belief, a_min=0, a_max=1)

    def compute_local_map_similarity_abs_err(self, pred_m, gt_m):
        if self.params['cmn_cfg']['local_map_type'] == "pred":
            return 1 - abs(pred_m.round() - gt_m.round()).sum() / (32 * 32)
        else:
            return np.array_equal(pred_m, gt_m) + 1e-2

    def render_local_map(self, loc):
        """ Render the ground truth local maps based on nearest neighbor metric"""
        # convert environment location to map location (Non-sampled map)
        if self.params['cmn_cfg']['rough_mpp'] == 0.01:
            c = int((loc[0] - self.env.min_x) // self.env.meters_per_pixel)
            r = int((loc[2] - self.env.min_y) // self.env.meters_per_pixel)
        else:
            c = int((loc[0] - self.env.min_x) // self.params['cmn_cfg']['rough_mpp']) + 1
            r = int((loc[2] - self.env.min_y) // self.params['cmn_cfg']['rough_mpp']) + 1

        # increase the index by 1 because we add a boundary outside the original grid
        r += 1
        c += 1

        # compute the l2 distance
        dist_list = [self.compute_l2_dist([r, c], loc_2) for loc_2 in self.rough_map.sampled_locations]

        # find the index of the map location
        local_map_id = dist_list.index(np.min(dist_list))

        # return the corresponding rough map
        local_map = self.rough_map.local_maps[local_map_id]

        # check the identity of local map
        assert local_map_id == local_map['id']
        if self.params['cmn_cfg']['local_map_type'] == "pred":
            return local_map['id'], local_map['center_loc_rough_map'], self.local_map_rgb2binary(local_map['map_arr'])
        else:
            return local_map['id'], local_map['center_loc_rough_map'], local_map['map_arr']

    def predict_local_map(self, obs):
        # obtain the observation
        rgb_obs_list, depth_obs_list = [], []
        if self.params['cmn_cfg']['obs_name'] == "color":
            rgb_obs_list = [obs['color_sensor_front'][:, :, 0:3],
                            obs['color_sensor_left'][:, :, 0:3],
                            obs['color_sensor_back'][:, :, 0:3],
                            obs['color_sensor_right'][:, :, 0:3]]
        elif self.params['cmn_cfg']['obs_name'] == "depth":
            depth_obs_list = [obs['depth_sensor_front'],
                              obs['depth_sensor_left'],
                              obs['depth_sensor_back'],
                              obs['depth_sensor_right']]
        elif self.params['cmn_cfg']['obs_name'] == "color_depth":
            rgb_obs_list = [obs['color_sensor_front'][:, :, 0:3],
                            obs['color_sensor_left'][:, :, 0:3],
                            obs['color_sensor_back'][:, :, 0:3],
                            obs['color_sensor_right'][:, :, 0:3]]
            depth_obs_list = [obs['depth_sensor_front'],
                              obs['depth_sensor_left'],
                              obs['depth_sensor_back'],
                              obs['depth_sensor_right']]
        else:
            raise Exception("Invalid observation type")

        # convert tensor
        if self.params['cmn_cfg']['obs_name'] == "color":
            obs_tensor = torch.tensor(np.array(rgb_obs_list)).float().permute(0, 3, 1, 2).to(self.device) / 255.0
        elif self.params['cmn_cfg']['obs_name'] == "depth":
            obs_tensor = torch.tensor(np.array(depth_obs_list)).float().unsqueeze(dim=1).to(self.device)
        elif self.params['cmn_cfg']['obs_name'] == "color_depth":
            tensor_rgb = torch.tensor(np.array(rgb_obs_list)).float().permute(0, 3, 1, 2).to(self.device) / 255.0
            tensor_depth = torch.tensor(np.array(depth_obs_list)).float().unsqueeze(dim=1).to(self.device)
            obs_tensor = torch.cat([tensor_rgb, tensor_depth], dim=1)
        else:
            raise Exception("Invalid observation type.")

        # predict the local map
        with torch.no_grad():
            # resize the observation
            obs_tensor = self.trans_resize(obs_tensor)
            # normalize the color observation
            if self.params['cmn_cfg']['obs_name'] == "color":
                obs_tensor = self.trans_normalize(obs_tensor)
            pred = self.local_map_predictor(obs_tensor).squeeze(dim=0).squeeze(dim=0).cpu().numpy()

        return pred

    def render_agent_direction(self, loc):
        """ Render the ground truth agent direction using odometry sensor data"""
        agent_map_angle = self.env.quaternion_to_angle(loc[3:])
        rot_control = int(np.round(agent_map_angle / (np.pi / 2)))
        if rot_control == 1:
            agent_dir = "east"
        elif rot_control == -1:
            agent_dir = "west"
        elif rot_control == 0:
            agent_dir = "south"
        else:
            agent_dir = "north"
        return agent_dir

    def estimate_agent_map_location(self):
        # find locations with maximal probability
        candidates = np.where(self.agent_belief_map == self.agent_belief_map.max())
        candidates = [[r, c] for r, c in zip(candidates[0].tolist(), candidates[1].tolist())]

        # find all candidates
        tmp_list = []
        for loc in self.rough_map.local_maps:
            if loc['loc_grid_map'] in candidates:
                tmp_list.append(loc)

        # randomly sample one and serve as the prediction
        if len(tmp_list) != 0:
            local_map = random.sample(tmp_list, 1)[0]
            local_map_loc = local_map['center_loc_rough_map']
            local_map_arr = local_map['map_arr']
            local_map_idx = local_map['id']
            return local_map_idx, local_map_loc, local_map_arr
        else:
            return None, None, None

    def belief_transition_func(self, agent_act, agent_dir):
        # pre-defined parameters for grid-based approach
        trans_dir_dict = {
            'east': {'shift': 1, 'axis': 1},
            'west': {'shift': -1, 'axis': 1},
            'north': {'shift': -1, 'axis': 0},
            'south': {'shift': 1, 'axis': 0}
        }
        shift = trans_dir_dict[agent_dir]['shift']
        axis = trans_dir_dict[agent_dir]['axis']

        # apply the move
        if agent_act == "move_forward":
            current_belief_copy = self.agent_belief_map.copy()
            # wall move into space area
            movable_locations = np.roll(self.wall_loc_map.copy(), shift=shift, axis=axis)
            # mask all wall locations
            movable_locations = np.multiply(self.wall_loc_map.copy(), movable_locations)
            # label the left adjacent locations with 1 and other space 2
            movable_locations = movable_locations + self.wall_loc_map
            wall_to_space_flags = np.where(movable_locations == 1.0, 1.0, 0.0)

            # space area move into walls
            movable_locations = np.roll(self.wall_loc_map.copy(), shift=-shift, axis=axis)
            # mask all wall locations
            movable_locations = np.multiply(self.wall_loc_map.copy(), movable_locations)
            # label the left adjacent locations with 1 and other space 2
            movable_locations = movable_locations + self.wall_loc_map
            space_to_wall_flags = np.where(movable_locations == 1.0, 1.0, 0.0)

            # update the belief
            if self.noise_trans_prob is None:
                pred_movable_belief = np.roll(current_belief_copy, shift=shift, axis=axis)
                pred_movable_belief = np.multiply(pred_movable_belief, self.wall_loc_map)
                prob_min_val = pred_movable_belief.min()
                pred_movable_belief = np.where(wall_to_space_flags == 1.0, prob_min_val, pred_movable_belief)

                # add the non-movable places
                pred_belief = pred_movable_belief + np.where(space_to_wall_flags == 1.0, current_belief_copy, 0)
            else:
                pred_movable_belief = np.roll(current_belief_copy, shift=shift, axis=axis)  # move the current belief
                pred_movable_belief = np.multiply(pred_movable_belief, self.wall_loc_map)  # mask the wall locations

                noise_trans_stay_prob = 1 - self.noise_trans_prob  # compute the stay probability
                noise_trans_stay_prob = np.multiply(noise_trans_stay_prob, self.wall_loc_map)  # mask the wall locations

                noise_trans_move_prob = np.roll(self.noise_trans_prob, shift=shift,
                                                axis=axis)  # compute the move probability
                noise_trans_move_prob = np.multiply(noise_trans_move_prob, self.wall_loc_map)  # mask the wall locations

                pred_belief = np.multiply(current_belief_copy, noise_trans_stay_prob) + np.multiply(pred_movable_belief,
                                                                                                    noise_trans_move_prob)
        else:
            pred_belief = self.agent_belief_map.copy()

        return pred_belief

    def observation_prob_func(self, obs):
        scores, directions = [], []
        estimated_loc_map = np.zeros_like(self.rough_map.map_sampled_grid)
        for m in self.rough_map.local_maps:
            # compute the similarity
            candidate_loc = m['loc_grid_map']
            if self.params['cmn_cfg']['local_map_type'] == "pred":
                candidate_map = self.local_map_rgb2binary(m['map_arr'])
            else:
                candidate_map = m['map_arr']
            # compute the similarity between predicted map and ground truth map
            # use absolute error
            score = self.compute_local_map_similarity_abs_err(obs, candidate_map)
            scores.append(score)

            # set the observation probability based on similarity
            estimated_loc_map[candidate_loc[0], candidate_loc[1]] = score

        return estimated_loc_map

    def add_noise_obs_conditioned(self, agent_dir):
        trans_dir_dict = {
            'east': {'shift': 1, 'axis': 1},
            'west': {'shift': -1, 'axis': 1},
            'north': {'shift': -1, 'axis': 0},
            'south': {'shift': 1, 'axis': 0}
        }
        shift = trans_dir_dict[agent_dir]['shift']
        axis = trans_dir_dict[agent_dir]['axis']

        # current state similarity
        current_patch_similarity = self.observation_prob_map.copy()

        # get the next state similarity
        next_patch_similarity = np.roll(current_patch_similarity, shift=-shift, axis=axis)
        next_patch_similarity = np.multiply(self.wall_loc_map, next_patch_similarity)
        current_patch_similarity = np.where(current_patch_similarity == 0.0, 1e-20, current_patch_similarity)

        # noisy transition probability
        noise_trans_prob = next_patch_similarity / (next_patch_similarity + current_patch_similarity)

        return noise_trans_prob

    def add_noise_random(self, agent_dir):
        trans_dir_dict = {
            'east': {'shift': 1, 'axis': 1},
            'west': {'shift': -1, 'axis': 1},
            'north': {'shift': -1, 'axis': 0},
            'south': {'shift': 1, 'axis': 0}
        }
        shift = trans_dir_dict[agent_dir]['shift']
        axis = trans_dir_dict[agent_dir]['axis']

        # model the transition function as a grid
        trans_func_grid = np.roll(self.wall_loc_map.copy(), shift=-shift, axis=axis)
        trans_func_grid = np.multiply(trans_func_grid, self.wall_loc_map)

        # create the noise trans prob
        noise_trans_prob = np.multiply(trans_func_grid, self.params['cmn_cfg']['add_noise_prob'])

        return noise_trans_prob

    # update the belief with noisy
    def noisy_update_belief(self, obs, pos, act):
        # compute the orientation of the agent, return is east, north, west, and south
        agent_dir = self.render_agent_direction(pos)

        if self.params['cmn_cfg']['local_map_type'] == "pred":  # if True, use the predicted local map
            # predict the local map
            map_obs = self.predict_local_map(obs)

            # rotate the local map
            if agent_dir == "east":
                map_obs = np.rot90(map_obs, k=1, axes=(1, 0))
                self.current_local_map = map_obs.copy()
            elif agent_dir == "north":
                self.current_local_map = map_obs.copy()
            elif agent_dir == "west":
                map_obs = np.rot90(map_obs, k=1)
                self.current_local_map = map_obs.copy()
            elif agent_dir == "south":
                map_obs = np.rot90(map_obs, k=2)
                self.current_local_map = map_obs.copy()
            else:
                raise Exception("Invalid agent direction")
        else:
            # use the ground truth local map
            _, _, map_obs = self.render_local_map(pos)
            map_obs = np.where(map_obs > 0.5, 1, map_obs)
            self.current_local_map = map_obs.copy()

        # compute the observation probability
        self.observation_prob_map = self.observation_prob_func(map_obs)

        # add noise
        if act == "move_forward":
            if self.params['cmn_cfg']['add_noise'] == "obs":
                self.noise_trans_prob = self.add_noise_obs_conditioned(agent_dir)
            elif self.params['cmn_cfg']['add_noise'] == "rnd":
                # randomly sample a p from a uniform distribution between [0, 1]
                self.params['cmn_cfg']['add_noise_prob'] = np.random.rand()
                self.noise_trans_prob = self.add_noise_random(agent_dir)
            elif self.params['cmn_cfg']['add_noise'] == "fixed":
                self.noise_trans_prob = self.add_noise_random(agent_dir)
            else:
                raise Exception("Invalid add noise mode")

        # compute the predictive belief
        self.predictive_belief_map = self.belief_transition_func(act, agent_dir)

        # correct the belief with observation probability
        log_belief_corrected = self.params['cmn_cfg']['obs_prob_weight'] * np.log(self.observation_prob_map) + \
                               self.params['cmn_cfg']['trans_prob_weight'] * np.log(self.predictive_belief_map)
        normalized_belief = np.exp(log_belief_corrected) / (np.exp(log_belief_corrected)).sum()

        # update the belief
        self.last_agent_belief_map = self.agent_belief_map.copy()
        self.updated_belief_map = normalized_belief.copy()
        self.agent_belief_map = normalized_belief.copy()

    def visualization(self, obs, t):
        # get the top-down view
        top_down_view = self.env.draw_agent_on_top_down_map()
        # plot policy and belief update
        if self.params['run_cfg']['visualization'] == "policy_and_belief":
            belief_map = self.normalize_belief_for_visualization(self.agent_belief_map.copy())
            if t == 0:
                self.sub_plots_arr[0].set_title("visualize policy")
                art_1 = self.sub_plots_arr[0].imshow(top_down_view)
                self.artists.append(art_1)
                self.sub_plots_arr[0].axis("off")
                self.sub_plots_arr[1].set_title("current belief")
                art_2 = self.sub_plots_arr[1].imshow(belief_map)
                self.artists.append(art_2)
                self.sub_plots_arr[1].axis("off")
            else:
                self.artists[0].set_data(top_down_view)
                self.artists[1].set_data(belief_map)
            self.vis_fig.canvas.draw()
            plt.pause(0.05)
        # plot belief update details
        elif self.params['run_cfg']['visualization'] == "belief_update":
            agent_belief = self.normalize_belief_for_visualization(self.last_agent_belief_map)
            pred_belief = self.normalize_belief_for_visualization(self.predictive_belief_map)
            obs_belief = self.normalize_belief_for_visualization(self.observation_prob_map)
            update_belief = self.normalize_belief_for_visualization(self.agent_belief_map)
            if t == 0:
                self.sub_plots_arr[0, 0].set_title("current belief")
                self.sub_plots_arr[0, 0].axis("off")
                art_1 = self.sub_plots_arr[0, 0].imshow(agent_belief)
                self.artists.append(art_1)
                self.sub_plots_arr[0, 1].set_title("trans pred belief")
                self.sub_plots_arr[0, 1].axis("off")
                art_2 = self.sub_plots_arr[0, 1].imshow(agent_belief)
                self.artists.append(art_2)
                self.sub_plots_arr[0, 2].set_title("obs prob")
                self.sub_plots_arr[0, 2].axis("off")
                art_3 = self.sub_plots_arr[0, 2].imshow(agent_belief)
                self.artists.append(art_3)
                self.sub_plots_arr[1, 0].set_title("updated belief")
                self.sub_plots_arr[1, 0].axis("off")
                art_4 = self.sub_plots_arr[1, 0].imshow(agent_belief)
                self.artists.append(art_4)
                self.sub_plots_arr[1, 1].set_title("local map est")
                self.sub_plots_arr[1, 1].axis("off")
                art_5 = self.sub_plots_arr[1, 1].imshow(self.current_local_map, vmin=0, vmax=1.0)
                self.artists.append(art_5)
                self.sub_plots_arr[1, 2].set_title("policy")
                art_6 = self.sub_plots_arr[1, 2].imshow(top_down_view)
                self.artists.append(art_6)
            else:
                self.artists[0].set_data(agent_belief)
                self.artists[1].set_data(pred_belief)
                self.artists[2].set_data(obs_belief)
                self.artists[3].set_data(update_belief)
                self.artists[4].set_data(self.current_local_map)
                self.artists[5].set_data(top_down_view)
            self.vis_fig.canvas.draw()
            plt.pause(0.05)
        else:
            self.env.render(obs, t)

        # save the figures
        # plt.savefig(f"/home/xcg/Desktop/NeurIPs_figures/visualizations/{env_name}/step_{t}.png", dpi=100)

    def get_action(self, obs):
        # if True, return a random policy
        if self.params['cmn_cfg']['enable_random_policy']:
            return random.sample(["move_forward", "turn_left", "turn_right"], 1)[0]

        # estimate agent location on the 2-D map using the current belief
        agent_map_idx, agent_map_loc, agent_local_map = self.estimate_agent_map_location()
        goal_map_idx, goal_map_loc, _ = self.render_local_map(self.env.goal_loc)

        # strategies for agent at goal location on the rough map no matter correct or wrong
        if agent_map_idx == goal_map_idx:
            # we will reset the belief to be the observation probability.
            self.agent_belief_map = self.observation_prob_map.copy()
            return random.sample(["move_forward", "turn_left", "turn_right"], 1)[0]

        # plan a path using Dijkstra's algorithm
        path = self.rough_map.dijkstra_path(agent_map_idx, goal_map_idx)

        if len(path) <= 1:
            return random.sample(["move_forward", "turn_left", "turn_right"], 1)[0]

        # compute the heuristic vector
        loc_1 = agent_map_loc
        loc_2 = self.rough_map.sampled_locations[path[1]]
        # reverse the location
        heu_vec = self.compute_norm_heuristic_vec([loc_1[1], loc_1[0]],
                                                  [loc_2[1], loc_2[0]])

        """ Given the heuristic vector, use a simple k-step ahead forward tree to find the best action
        """
        # do a k-step breadth first search
        # todo: mpp_01_to_05 we use the relative location info but still the absolute orientation value
        if self.params['cmn_cfg']['use_pos_ori']:
            tmp_current_state = np.array(obs['state'][0:3])
            tmp_goal_state = self.env.goal_loc
            relative_location = (tmp_goal_state - tmp_current_state).tolist() + obs['state'][3:]
        else:
            relative_location = [0, 0, 0] + obs['state'][3:]

        root_node = TreeNode(relative_location)
        self.breadth_first_tree = BFTree(root_node,
                                         depth=self.params['cmn_cfg']['forward_step_k'],
                                         agent_forward_step=self.params['agent_cfg']['move_forward'])
        best_child_node = self.breadth_first_tree.find_the_best_child(heu_vec)

        # retrieve the tree to get the action
        parent = best_child_node.parent
        while parent.parent is not None:
            best_child_node = best_child_node.parent
            parent = best_child_node.parent

        # based on the results select the best action
        return best_child_node.name

    def is_navigatable_env(self, goal_loc):
        # sample start and goals that can be used to find the optimal path
        try:
            actions = self.env.follower.find_path(goal_loc)
        except habitat_sim.errors.GreedyFollowerError:
            actions = None
        return actions

    def is_navigatable_map(self, start_loc, goal_loc):
        # copy the grid to plan a path
        grid = self.rough_map.map_binary_arr.copy()

        # compute the start and goal location on the map
        start_loc = self.render_local_map(start_loc)[1]
        goal_loc = self.render_local_map(goal_loc)[1]

        # check if the goal exists on the map
        if grid[goal_loc[0], goal_loc[1]]:
            return False

        # check if there exists a path between start location and goal location
        path = A_star(grid, start_loc, goal_loc)
        return path is not None

    def run(self, name, env):
        # save the environment
        self.env_name = name
        self.env = env

        # create the graph-structured 2-D coarse map
        self.init_graph_coarse_map()

        # SPL and SR
        spl = 0
        sr = 0
        e_counter = 0
        # run evaluation
        episode_num = self.params['run_cfg']['episode_num']
        while e_counter < episode_num:
            """ initialization
            """
            # init the beliefs
            self.init_belief()
            # reset the environment
            obs = self.env.reset()
            self.get_action(obs)

            """ Sample start and goal locations
            """
            # sampled start and goal positions
            start_pos = self.env.start_loc
            goal_pos = self.env.goal_loc

            # obtain the optimal actions planned from the environment
            optimal_actions = self.is_navigatable_env(goal_pos)

            # drop the pair if there is no path between the start and goal locations in the environment
            if optimal_actions is None:
                continue

            # drop the pair if there is no path between the start and goal locations on the map
            if not self.is_navigatable_map(start_pos, goal_pos):
                continue

            # visualization
            if self.params['run_cfg']['show_visualization']:
                self.visualization(obs, e_counter)

            # remove the last None element
            optimal_actions.pop()

            # reset the geometric distance
            geo_dist = 0
            for act in optimal_actions:
                if act == "move_forward":
                    geo_dist += self.params['agent_cfg']['move_forward']
                else:
                    continue
            env.geo_dist = geo_dist
            if env.geo_dist == 0.0:
                continue

            # increase the episode counter
            e_counter += 1
            episode_time_step = self.params['run_cfg']['nav_episode_budget']
            print(f"-----------------------------------------------------------------")
            print(f"Episode idx = {e_counter}: "
                  f"start loc = {np.round(start_pos, 2)}, goal loc = {np.round(goal_pos, 2)} starts...")
            """ Add logging
            """
            # add the information to log
            if self.params['run_cfg']['enable_logging']:
                logging.info(f"Episode idx = {e_counter},"
                             f" start loc = {np.round(start_pos, 2)},"
                             f" goal loc = {np.round(goal_pos, 2)} starts...")

            # compute the relative goal position
            relative_goal_pos = goal_pos - start_pos
            relative_error = np.linalg.norm(goal_pos - start_pos)
            # statistics variables for current episode
            success_flag = False
            nav_dist = 0
            nav_step = 0
            # obtain the optimal actions
            real_actions = []
            for t in range(episode_time_step):
                # plan an action
                if self.params['cmn_cfg']['enable_optimal_policy']:
                    act = optimal_actions[t]
                else:
                    act = self.get_action(obs)
                real_actions.append(act)

                # step in the env
                next_obs, done = self.env.step(act)
                nav_step += 1

                # compute the cumulative distance
                if act == "move_forward":
                    nav_dist += self.params['agent_cfg']['move_forward']

                # update the belief
                if not self.params['cmn_cfg']['enable_random_policy']:
                    self.noisy_update_belief(obs, obs['state'], act)

                # print current agent location information
                print(f"{e_counter}: time step = {t}, "
                      f"loc = {obs['state']}, action = {act}, next loc = {next_obs['state']}, "
                      f"dist={relative_error}")

                # visualization
                if self.params['run_cfg']['show_visualization']:
                    self.visualization(obs, t + e_counter + 1)

                # compute the current relative position
                current_relative_pos = next_obs['state'][0:3] - start_pos

                # compute the distance
                relative_error = np.linalg.norm(current_relative_pos - relative_goal_pos)

                # check termination
                if relative_error <= self.env.dist_eps:
                    success_flag = True  # if True
                    sr += 1  # count by 1
                    spl += self.env.geo_dist / max(nav_dist, self.env.geo_dist)
                    break  # end the episode

                # increment the observation
                obs = next_obs

            # compute the final distance between start and goal
            end_loc = obs['state'][0:3]
            dist_to_go = self.env.compute_geo_distance(obs['state'][0:3], self.env.goal_loc)
            # add the information to log
            if success_flag:
                print(f"Episode idx = {e_counter}, start loc = {np.round(start_pos, 2)},"
                      f" goal loc = {np.round(goal_pos, 2)} SUCCESS:)")
                if self.params['run_cfg']['enable_logging']:
                    logging.info(
                        f"Episode idx = {e_counter}, end loc = {np.round(end_loc, 2)}, goal loc = {np.round(goal_pos, 2)} SUCCESS:)")
                    logging.info(
                        f"Start geo distance = {np.round(self.env.geo_dist, 2)}, End geo distance = {np.round(dist_to_go, 2)}")
                    logging.info(f"Dist SPL metric = {self.env.geo_dist / max(nav_dist, self.env.geo_dist)}")
                    logging.info(
                        f"Act SPL metric = {len(optimal_actions) / max(len(real_actions), len(optimal_actions))}")
                    logging.info(f"Optimal actions = {optimal_actions}")
                    logging.info(f"Real actions = {real_actions}")
                    logging.info(f"-------------------------------------------------------------------------")
                print("--------------------------------------------------------------------------")
            else:
                print(f"Episode idx = {e_counter}, start loc = {np.round(start_pos, 2)},"
                      f" goal loc = {np.round(goal_pos, 2)} FAIL:(")
                if self.params['run_cfg']['enable_logging']:
                    logging.info(
                        f"Episode idx = {e_counter}, end loc = {np.round(end_loc, 2)}, goal loc = {np.round(goal_pos, 2)} FAIL:(")
                    logging.info(
                        f"Start geo distance = {np.round(self.env.geo_dist, 2)}, End geo distance = {np.round(dist_to_go, 2)}")
                    # logging.info(f"Loc error = {str(episodic_loc_error)}")
                    logging.info(f"Optimal actions = {optimal_actions}")
                    logging.info(f"Real actions = {real_actions}")
                    logging.info(f"-------------------------------------------------------------------------")
                print("--------------------------------------------------------------------------")

            print(f"Current success rate = {sr / e_counter}")
            print(f"Current distance spl = {spl / e_counter}")

        print("**************************************")
        print(f"The mean success rate = {sr / episode_num}")
        print(f"The mean distance spl = {spl / episode_num}")
        print("**************************************")

        if self.params['run_cfg']['enable_logging']:
            logging.info("**************************************")
            logging.info(f"The mean success rate = {sr / episode_num}")
            logging.info(f"The mean distance spl = {spl / episode_num}")
            logging.info("**************************************")

        return sr / episode_num, spl / episode_num


if __name__ == "__main__":
    # load CMN parameters
    general_params = ParamsLoader("params/param_run_baseline_uniform_belief_rescaled_hand_drawn.json").params_data

    # set the randomness for reproduction
    np.random.seed(general_params['run_cfg']['random_seed'])
    random.seed(general_params['run_cfg']['random_seed'])
    torch.manual_seed(general_params['run_cfg']['random_seed'])

    # loaded the pre-trained deep local map predictor
    general_params['cmn_cfg']['local_map_predictor_model'] = f"results/gibson_split" \
                                                             f"/run_{int(general_params['run_cfg']['random_seed'] // 10)}_seed={general_params['run_cfg']['random_seed']}" \
                                                             f"/gibson/{general_params['cmn_cfg']['obs_name']}" \
                                                             f"/dataset=gibson_" \
                                                             f"rnd_seed={general_params['run_cfg']['random_seed']}_" \
                                                             f"obs={general_params['cmn_cfg']['obs_name']}_" \
                                                             f"bs=32_lr=0.0001_wd=1e-07_dp=0.5_pretrained=1_" \
                                                             f"local_map_predictor_val_eval.pt"

    # create the CMN navigator object
    cmn = CoarseMapNavigator(general_params)

    # load environments to evaluate
    env_names = load_environment_names(general_params['run_cfg']['dataset_name'],
                                       general_params['run_cfg']['mode'],
                                       general_params['run_cfg']['random_seed'])

    # save the logs
    if general_params['run_cfg']["enable_logging"]:
        # create saving directories
        if not os.path.exists(general_params['run_cfg']['log_save_path']):
            os.makedirs(general_params['run_cfg']['log_save_path'])

        # create the file name
        log_file_name = f"{general_params['run_cfg']['log_save_path']}/run_seed={general_params['run_cfg']['run_num']}.log"

        # create the log to save data
        if os.path.exists(log_file_name):
            os.remove(log_file_name)
        logging.basicConfig(filename=log_file_name, level=logging.DEBUG)

    # run evaluation
    env_results = {}
    for idx, env_name in enumerate(env_names):
        if general_params['run_cfg']['enable_logging']:
            logging.info(f"******* Evaluation run seed = {general_params['run_cfg']['random_seed']} starts  *******")
            logging.info(f"******* Env name = {env_name}  *******")

        # create the environment
        my_house = make_env(env_name, general_params)

        # plan in the environment
        env_sr, env_spl = cmn.run(env_name, my_house)

        # append the results
        env_res = {'sr': env_sr, 'spl': env_spl}
        env_results[env_name] = env_res

        # close the environment
        my_house.close()

        break
