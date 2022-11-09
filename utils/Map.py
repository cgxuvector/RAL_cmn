from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class TopoMap(object):
    def __init__(self, map_arr, configs):
        # convert the RGB occupancy map to binary representation
        if len(map_arr.shape) == 3:
            map_arr = rgb2gray(map_arr)

        # binary map array
        self.map_row, self.map_col = map_arr.shape
        self.map_binary_arr = np.ones((self.map_row+2, self.map_col+2))
        self.map_binary_arr[1:self.map_row+1, 1:self.map_col+1] = map_arr
        self.map_row, self.map_col = self.map_binary_arr.shape
        self.map_binary_arr = np.where(self.map_binary_arr < 0.5, 0, 1.0)
        self.local_maps = self.get_valid_locations(False)

        # map configurations
        self.map_configs = configs

        # make the graph
        self.global_map_graph, self.global_map_dict, self.sampled_locations = self.make_graph()

        # use the rough map as the map sampled grid
        self.map_sampled_grid = 1 - self.map_binary_arr.copy()

        # set the local map size
        self.local_map_size = [configs['coarse_local_map_size'],
                               configs['coarse_local_map_size']]

    def get_valid_locations(self, display=True):
        loc_coords = np.where(self.map_binary_arr == 0.0)
        space_locs = [(r, c) for r, c in zip(loc_coords[0], loc_coords[1])]

        # whether show the cropping process
        if display:
            fig, arrs = plt.subplots(1, 2)
            artist_1, artist_2 = 0, 0
            init_artists = False

        # crop local maps from the coarse map with size self.local_map_size
        cropped_local_maps = []
        for id_counter, space in enumerate(space_locs):
            # remove free spaces on the edges
            if (0 < space[0] < (self.map_row - 1)) and (0 < space[1] < (self.map_col - 1)):
                # crop the local map
                from_r = space[0] - 1
                to_r = space[0] + 2
                from_c = space[1] - 1
                to_c = space[1] + 2
                local_map = self.map_binary_arr[from_r:to_r, from_c:to_c]

                # save the local maps
                cropped_local_maps.append({'id': id_counter,  # local map index
                                           'center_loc_rough_map': space,  # local map center location on rough map
                                           'loc_grid_map': [space[0], space[1]],  # location on the sampled grid map
                                           'map_arr': local_map  # local map array
                                           })
                id_counter += 1

                # draw local map on global map
                res_env_map = self.map_binary_arr.copy()
                res_env_map[from_r:to_r, from_c:to_c] = local_map + 0.1

                if display:
                    # plot the local map and global
                    if not init_artists:
                        artist_1 = arrs[0].imshow(res_env_map, cmap='gray')
                        arrs[1].set_title(f"({space})")
                        artist_2 = arrs[1].imshow(local_map, cmap='gray')
                        init_artists = True
                    else:
                        artist_1.set_data(res_env_map)
                        artist_2.set_data(local_map)
                    fig.canvas.draw()
                    plt.pause(0.01)

        if display:
            plt.close(fig)

        return cropped_local_maps

    # function is used to make the global graph
    def make_graph(self):
        # boundary
        min_r, max_r = 0, self.map_binary_arr.shape[0] - 1
        min_c, max_c = 0, self.map_binary_arr.shape[1] - 1

        # find the neighbors
        def find_neighbors(center_loc, act_stride=1):
            # actions
            actions = [[-act_stride, 0], [act_stride, 0], [0, -act_stride], [0, act_stride]]
            # compute the neighbors
            neighbor_idx = []
            for act in actions:
                # compute the neighbor location and clip the invalid value
                neighbor_loc_r = center_loc[0] + act[0]
                neighbor_loc_c = center_loc[1] + act[1]

                # check the validation
                if neighbor_loc_c > max_c or neighbor_loc_c < min_c or neighbor_loc_r > max_r or neighbor_loc_r < min_r:
                    continue
                neighbor_loc = (neighbor_loc_r, neighbor_loc_c)

                if neighbor_loc in sampled_cells:
                    neighbor_idx.append(sampled_cells.index(neighbor_loc))

            return neighbor_idx

        # find the corridor cells
        sampled_cells = [item['center_loc_rough_map'] for item in self.local_maps]

        # build the graph dict
        graph_dict = {}
        for idx, loc in enumerate(sampled_cells):
            # create the dict for current vertex
            vertex_dict = {'loc': loc, 'edges': []}
            # find the indices of all neighbors
            neighbor_indices = find_neighbors(loc)
            # add all edges to the vertex
            for n_idx in neighbor_indices:
                vertex_dict['edges'].append((idx, n_idx))
            # save the vertex and its edges
            graph_dict[idx] = vertex_dict

        # build the graph from the dict
        graph = self.make_graph_from_dict(graph_dict)

        return graph, graph_dict, sampled_cells

    # function is used to plan the shortest path between two vertex using dijkstra's algorithm
    def dijkstra_path(self, s_node, e_node):
        # find the shortest path
        try:
            shortest_path = nx.dijkstra_path(self.global_map_graph, s_node, e_node)
        except nx.NetworkXNoPath:
            shortest_path = []

        return shortest_path

    def display_shortest_path_on_map(self, shortest_path):
        # array for visualization
        vis_arr = self.map_binary_arr.copy()

        # color the way point from start to goal with darker
        path_val_list = np.linspace(0.1, 0.9, len(shortest_path)).tolist()

        # loop all the locations in the path
        for idx, vertex in enumerate(shortest_path):
            loc = self.global_map_graph.nodes[vertex]['loc']
            if idx == 0:  # plot the start
                val = 0.1
            elif idx == len(shortest_path) - 1:  # plot the goal
                val = 0.9
            else:  # plot the points in the middle
                val = path_val_list[idx]

            #todo: comment this after debug
            vis_arr[loc[0], loc[1]] = val

            #todo: uncomment this after debug
            # # draw lines between locations
            # if idx != len(shortest_path) - 1:
            #     start_loc = loc
            #     end_loc = self.global_map_graph.nodes[shortest_path[idx+1]]['loc']
            #     plt.plot([start_loc[1], end_loc[1]],
            #              [start_loc[0], end_loc[0]],
            #              linewidth=2,
            #              color="red",
            #              alpha=val)
            #     plt.plot(start_loc[1], start_loc[0], "ro", alpha=val)

        # plt.plot(end_loc[1], end_loc[0], "ro", alpha=val)
        # plt.imshow(1 - vis_arr, cmap="gray")
        # plt.show()
        return 1 - vis_arr

    def display_graph_on_map(self):
        # extract all nodes
        vertices = [self.global_map_dict[key]['loc'] for key in self.global_map_dict.keys()]
        edges = [self.global_map_dict[key]['edges'] for key in self.global_map_dict.keys()]

        # plot the vertices on the map with size 10  x 10 in pixels
        display_map = self.map_binary_arr.copy()

        # plot the edges on the map
        drawn_pair = []
        for idx, item in enumerate(edges):
            print(f"{idx}-{len(edges)}: {item}")
            for sub_item in item:
                if sub_item in drawn_pair:
                    continue
                else:
                    # start and goal locations
                    start_loc = self.sampled_locations[sub_item[0]]
                    end_loc = self.sampled_locations[sub_item[1]]

                    # align the coordinates
                    line_from = [start_loc[1], end_loc[1]]
                    line_to = [start_loc[0], end_loc[0]]
                    plt.plot(start_loc[1], start_loc[0], "ro")
                    plt.plot(end_loc[1], end_loc[0], 'ro')
                    plt.plot(line_from, line_to, linewidth=1, color='red')

                    # save the drawn pairs
                    drawn_pair.append(sub_item)
                    drawn_pair.append((sub_item[1], sub_item[0]))

        # plot the results
        plt.title(f"Local map size = {self.map_configs['local_map_size'] * 2}")
        plt.imshow(1 - display_map, cmap="gray")
        plt.show()

    @staticmethod
    def make_graph_from_dict(g_dict):
        G = nx.Graph()
        for key, val in g_dict.items():
            # add the node
            G.add_node(key, loc=val['loc'])
            # add the edges
            for e in val['edges']:
                G.add_edge(*e, start=val['loc'], end=e)
        return G

    @staticmethod
    def compute_dist(s_loc, e_loc):
        s_loc = np.array(s_loc)
        e_loc = np.array(e_loc)
        dist = np.linalg.norm(s_loc - e_loc)
        return dist