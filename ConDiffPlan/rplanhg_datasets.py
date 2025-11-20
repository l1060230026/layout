import math
import random
import torch as th

from PIL import Image, ImageDraw
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from glob import glob
import json
import os
import cv2 as cv
from tqdm import tqdm
from shapely import geometry as gm
from shapely.ops import unary_union
from collections import defaultdict
import copy
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt

def visualize_polygon(polygons):
    """
    Visualize the vertices of a given closed polygon.
    
    Parameters:
    vertices (list of list): List of polygon vertices, format: [[x1, y1], [x2, y2], ...]
    """
    
    plt.figure(figsize=(10, 10))
    
    for vertices in polygons:
        # Convert vertices to numpy array
        vertices = np.array(vertices)
        
        # Ensure polygon is closed
        vertices = np.vstack([vertices, vertices[0]])  # Add first point to the end to close the polygon

        # Draw polygon, increase line width, remove point markers
        plt.plot(vertices[:, 0], vertices[:, 1], linewidth=2)

    plt.axis('off')  # Hide axes
    plt.axis('equal')  # Keep aspect ratio
    plt.show()


def load_rplanhg_data(
    batch_size,
    analog_bit,
    target_set = 8,
    set_name = 'train',
    rtype_dim=25,
    corner_index_dim=32,
    room_index_dim=32,
    max_num_points=200,
):
    """
    For a dataset, create a generator over (shapes, kwargs) pairs.
    
    Args:
        batch_size: Batch size for data loading
        analog_bit: Whether to use analog bit encoding
        target_set: Target set number
        set_name: 'train' or 'eval'
        rtype_dim: Dimension for room type one-hot encoding (default: 25)
        corner_index_dim: Max corners per room & dimension for corner index one-hot encoding (default: 32)
        room_index_dim: Max rooms per house & dimension for room index one-hot encoding (default: 32)
        max_num_points: Maximum number of points in a house layout (default: 200)
    """
    print(f"loading {set_name} of target set {target_set}")
    deterministic = False if set_name=='train' else True
    dataset = RPlanhgDataset(set_name, analog_bit, target_set, non_manhattan=False,
                             rtype_dim=rtype_dim,
                             corner_index_dim=corner_index_dim,
                             room_index_dim=room_index_dim,
                             max_num_points=max_num_points)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
        )
    while True:
        yield from loader


def make_non_manhattan(poly, polygon, house_poly):
    dist = abs(poly[2]-poly[0])
    direction = np.argmin(dist)
    center = poly.mean(0)
    min = poly.min(0)
    max = poly.max(0)

    tmp = np.random.randint(3, 7)
    new_min_y = center[1]-(max[1]-min[1])/tmp
    new_max_y = center[1]+(max[1]-min[1])/tmp
    if center[0]<128:
        new_min_x = min[0]-(max[0]-min[0])/np.random.randint(2,5)
        new_max_x = center[0]
        poly1=[[min[0], min[1]], [new_min_x, new_min_y], [new_min_x, new_max_y], [min[0], max[1]], [max[0], max[1]], [max[0], min[1]]]
    else:
        new_min_x = center[0]
        new_max_x = max[0]+(max[0]-min[0])/np.random.randint(2,5)
        poly1=[[min[0], min[1]], [min[0], max[1]], [max[0], max[1]], [new_max_x, new_max_y], [new_max_x, new_min_y], [max[0], min[1]]]

    new_min_x = center[0]-(max[0]-min[0])/tmp
    new_max_x = center[0]+(max[0]-min[0])/tmp
    if center[1]<128:
        new_min_y = min[1]-(max[1]-min[1])/np.random.randint(2,5)
        new_max_y = center[1]
        poly2=[[min[0], min[1]], [min[0], max[1]], [max[0], max[1]], [max[0], min[1]], [new_max_x, new_min_y], [new_min_x, new_min_y]]
    else:
        new_min_y = center[1]
        new_max_y = max[1]+(max[1]-min[1])/np.random.randint(2,5)
        poly2=[[min[0], min[1]], [min[0], max[1]], [new_min_x, new_max_y], [new_max_x, new_max_y], [max[0], max[1]], [max[0], min[1]]]
    p1 = gm.Polygon(poly1)
    iou1 = house_poly.intersection(p1).area/ p1.area
    p2 = gm.Polygon(poly2)
    iou2 = house_poly.intersection(p2).area/ p2.area
    if iou1>0.9 and iou2>0.9:
        return poly
    if iou1<iou2:
        return poly1
    else:
        return poly2

def aug_points(arr, step=9):
    t_values = np.linspace(0, 1, step)
    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 2]
    y2 = arr[:, 3]

    # Calculate coordinate increments
    delta_x = x2 - x1
    delta_y = y2 - y1

    # Calculate x and y coordinates for each t
    xs = x1[:, np.newaxis] + delta_x[:, np.newaxis] * t_values[np.newaxis, :]
    ys = y1[:, np.newaxis] + delta_y[:, np.newaxis] * t_values[np.newaxis, :]

    # Merge coordinates and adjust shape
    points = np.stack([xs, ys], axis=2)
    result = points.reshape(-1, step * 2)
    return result

get_bin = lambda x, z: [int(y) for y in format(x, 'b').zfill(z)]
get_one_hot = lambda x, z: np.eye(z)[x]
class RPlanhgDataset(Dataset):
    def __init__(self, set_name, analog_bit, target_set, non_manhattan=False,
                 rtype_dim=25, corner_index_dim=32, room_index_dim=32,
                 max_num_points=100):
        super().__init__()
        base_dir = './datasets/rplan'
        self.non_manhattan = non_manhattan
        self.set_name = set_name
        self.analog_bit = analog_bit
        self.target_set = target_set
        
        # Hyperparameters
        self.rtype_dim = rtype_dim
        self.corner_index_dim = corner_index_dim  # also defines max corners per room
        self.room_index_dim = room_index_dim      # also defines max rooms per house
        self.max_num_points = max_num_points
        
        self.subgraphs = []
        self.org_graphs = []
        self.org_houses = []
        max_num_points = self.max_num_points
        if self.set_name == 'eval':
            cnumber_dist = np.load(f'processed_rplan/rplan_train_{target_set}_cndist.npz', allow_pickle=True)['cnumber_dist'].item()
        if not True:
        # if self.set_name == 'train' and os.path.exists(f'processed_rplan/rplan_{set_name}_{target_set}.npz'):
                data = np.load(f'processed_rplan/rplan_{set_name}_{target_set}.npz', allow_pickle=True)
                self.graphs = data['graphs']
                self.houses = data['houses']
                self.boundarys = data['boundarys']
                self.door_masks = data['door_masks']
                self.self_masks = data['self_masks']
                self.gen_masks = data['gen_masks']
                self.bound_masks = data['bound_masks']
                self.num_coords = 2
                self.max_num_points = max_num_points
                cnumber_dist = np.load(f'processed_rplan/rplan_train_{target_set}_cndist.npz', allow_pickle=True)['cnumber_dist'].item()
        else:
            with open(f'{base_dir}/list.txt') as f:
                lines = f.readlines()
            cnt=0
            for line in tqdm(lines):
                cnt=cnt+1
                file_name = f'{base_dir}/{line[:-1]}'
                rms_type,graph,r_boundarys=reader(file_name)
                
                fp_size = len([x for x in rms_type if x != 15 and x != 17 and x != 18])
                graph_nodes, graph_edges = self.build_graph(rms_type, r_boundarys, graph)

                skip_file = False
                room_boundaries = []
                for room_mask in r_boundarys:

                    if len(room_mask) > self.corner_index_dim:  # Skip entire file if overly long contour is found
                        skip_file = True
                        break
                    room_boundaries.append(room_mask)
                
                if skip_file:
                    continue  # Skip current file directly
                    
                if self.set_name=='train' and fp_size not in [5, 6, 7]:
                        continue
                if self.set_name=='eval' and fp_size != target_set:
                # if self.set_name=='eval' and fp_size not in [5, 6, 7]:
                        continue
                a = [rms_type, room_boundaries,]
                self.subgraphs.append(a)
                
                house = []
                for index, room_type in enumerate(rms_type):
                    contours = room_boundaries[index]
                    house.append([contours, room_type])
                self.org_graphs.append(graph_edges)
                self.org_houses.append(house)
                    
                # if len(self.subgraphs) == 1000:
                #     break
                
            houses = []
            door_masks = []
            self_masks = []
            gen_masks = []
            graphs = []
            if self.set_name=='train':
                cnumber_dist = defaultdict(list)

            if self.non_manhattan:
                for h, graph in tqdm(zip(self.org_houses, self.org_graphs), desc='processing dataset'):
                    # Generating non-manhattan Balconies
                    tmp = []
                    for i, room in enumerate(h):
                        if room[1]>10:
                            continue
                        if len(room[0])!=4: 
                            continue
                        if np.random.randint(2):
                            continue
                        poly = gm.Polygon(room[0])
                        house_polygon = unary_union([gm.Polygon(room[0]) for room in h])
                        room[0] = make_non_manhattan(room[0], poly, house_polygon)

            for index, (h, graph) in tqdm(enumerate(zip(self.org_houses, self.org_graphs)), total=len(self.org_houses), desc='processing dataset'):
                house = []
                corner_bounds = []
                num_points = 0
                for i, room in enumerate(h):
                    # if room[1]>10:
                    #     room[1] = {15:11, 17:12, 16:13}[room[1]]
                    room[0] = np.reshape(room[0], [len(room[0]), 2])/256. - 0.5 # [[x0,y0],[x1,y1],...,[x15,y15]] and map to 0-1 - > -0.5, 0.5
                    room[0] = room[0] * 2 # map to [-1, 1]
                    if self.set_name=='train':
                        cnumber_dist[room[1]].append(len(room[0]))
                    # Adding conditions
                    num_room_corners = len(room[0])
                    rtype = np.repeat(np.array([get_one_hot(room[1]+1, self.rtype_dim)]), num_room_corners, 0)
                    room_index = np.repeat(np.array([get_one_hot(len(house)+1, self.room_index_dim)]), num_room_corners, 0)
                    corner_index = np.array([get_one_hot(x, self.corner_index_dim) for x in range(num_room_corners)])
                    # Src_key_padding_mask
                    if room[1] in [15, 18]:
                        padding_mask = np.repeat(0, num_room_corners)
                        padding_mask = np.expand_dims(padding_mask, 1)
                        fixed_mask = np.repeat(np.array([[1]]), num_room_corners, 0)
                    else:
                        padding_mask = np.repeat(1, num_room_corners)
                        padding_mask = np.expand_dims(padding_mask, 1)
                        fixed_mask = np.repeat(np.array([[0]]), num_room_corners, 0)
                    # Generating corner bounds for attention masks
                    connections = np.array([[i,(i+1)%num_room_corners] for i in range(num_room_corners)])
                    connections += num_points
                    corner_bounds.append([num_points, num_points+num_room_corners])
                    num_points += num_room_corners
                    room = np.concatenate((room[0], rtype, corner_index, room_index, padding_mask, connections, fixed_mask), 1)
                    house.append(room)

                house_layouts = np.concatenate(house, 0)
                if len(house_layouts)>max_num_points:
                    continue
                # Calculate feature dimension: coords + rtype + corner_index + room_index + padding_mask + connections + fixed_mask
                feature_dim = 2 + self.rtype_dim + self.corner_index_dim + self.room_index_dim + 1 + 2 + 1
                padding = np.zeros((max_num_points-len(house_layouts), feature_dim))
                gen_mask = np.ones((max_num_points, max_num_points))
                gen_mask[:len(house_layouts), :len(house_layouts)] = 0
                house_layouts = np.concatenate((house_layouts, padding), 0)

                door_mask = np.ones((max_num_points, max_num_points)) # Connected rooms
                self_mask = np.ones((max_num_points, max_num_points)) # Within the same room
                for i in range(len(corner_bounds)):
                    for j in range(len(corner_bounds)):
                        if i==j:
                            self_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                        elif any(np.equal([i, 1, j], graph).all(1)) or any(np.equal([j, 1, i], graph).all(1)):
                            door_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                houses.append(house_layouts)
                door_masks.append(door_mask)
                self_masks.append(self_mask)
                gen_masks.append(gen_mask)
                graphs.append(graph)
            self.max_num_points = max_num_points
            self.houses = houses
            self.door_masks = door_masks
            self.self_masks = self_masks
            self.gen_masks = gen_masks
            self.num_coords = 2
            self.graphs = graphs

            # np.savez_compressed(f'processed_rplan/rplan_{set_name}_{target_set}', graphs=self.graphs, houses=self.houses, boundarys=self.boundarys,
                    # door_masks=self.door_masks, self_masks=self.self_masks, gen_masks=self.gen_masks, bound_masks=self.bound_masks)
            if self.set_name=='train':
                np.savez_compressed(f'processed_rplan/rplan_{set_name}_{target_set}_cndist', cnumber_dist=cnumber_dist)

            if set_name=='eval':
                houses = []
                graphs = []
                door_masks = []
                self_masks = []
                gen_masks = []
                len_house_layouts = 0
                for index, (h, graph) in tqdm(enumerate(zip(self.org_houses, self.org_graphs)), desc='processing syn dataset', total=len(self.org_houses)):
                    house = []
                    corner_bounds = []
                    num_points = 0
                    # num_room_corners_total = [len(room[0]) for room in h]
                    num_room_corners_total = [cnumber_dist[room[1]][random.randint(0, len(cnumber_dist[room[1]])-1)] for room in h]
                    num_room_corners_total[-1] = len(h[-1][0])
                    while np.sum(num_room_corners_total)>=max_num_points:
                        num_room_corners_total = [cnumber_dist[room[1]][random.randint(0, len(cnumber_dist[room[1]])-1)] for room in h]
                        num_room_corners_total[-1] = len(h[-1][0])
                    for i, room in enumerate(h):
                        # Adding conditions
                        num_room_corners = num_room_corners_total[i]
                        rtype = np.repeat(np.array([get_one_hot(room[1]+1, self.rtype_dim)]), num_room_corners, 0)
                        room_index = np.repeat(np.array([get_one_hot(len(house)+1, self.room_index_dim)]), num_room_corners, 0)
                        corner_index = np.array([get_one_hot(x, self.corner_index_dim) for x in range(num_room_corners)])
                        # Src_key_padding_mask
                        if room[1] in [15, 18]:
                            padding_mask = np.repeat(0, num_room_corners)
                            padding_mask = np.expand_dims(padding_mask, 1)
                            fixed_mask = np.repeat(np.array([[1]]), num_room_corners, 0)
                        else:
                            padding_mask = np.repeat(1, num_room_corners)
                            padding_mask = np.expand_dims(padding_mask, 1)
                            fixed_mask = np.repeat(np.array([[0]]), num_room_corners, 0)
                        # Generating corner bounds for attention masks
                        connections = np.array([[i,(i+1)%num_room_corners] for i in range(num_room_corners)])
                        connections += num_points
                        corner_bounds.append([num_points, num_points+num_room_corners])
                        num_points += num_room_corners
                        if room[1] not in [15, 18]:
                            room = np.concatenate((np.zeros([num_room_corners, 2]), rtype, corner_index, room_index, padding_mask, connections, fixed_mask), 1)
                        else:
                            room = np.concatenate((room[0], rtype, corner_index, room_index, padding_mask, connections, fixed_mask), 1)
                        house.append(room)

                    house_layouts = np.concatenate(house, 0)
                    if np.sum([len(room[0]) for room in h])>max_num_points:
                        continue
                        
                    # Calculate feature dimension: coords + rtype + corner_index + room_index + padding_mask + connections + fixed_mask
                    feature_dim = 2 + self.rtype_dim + self.corner_index_dim + self.room_index_dim + 1 + 2 + 1
                    padding = np.zeros((max_num_points-len(house_layouts), feature_dim))
                    gen_mask = np.ones((max_num_points, max_num_points))
                    gen_mask[:len(house_layouts), :len(house_layouts)] = 0
                    house_layouts = np.concatenate((house_layouts, padding), 0)

                    door_mask = np.ones((max_num_points, max_num_points))
                    self_mask = np.ones((max_num_points, max_num_points))
                    
                    for i in range(len(corner_bounds)):
                        for j in range(len(corner_bounds)):
                            if i==j:
                                self_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0
                            elif any(np.equal([i, 1, j], graph).all(1)) or any(np.equal([j, 1, i], graph).all(1)):
                                door_mask[corner_bounds[i][0]:corner_bounds[i][1],corner_bounds[j][0]:corner_bounds[j][1]] = 0

                    houses.append(house_layouts)
                    door_masks.append(door_mask)
                    self_masks.append(self_mask)
                    gen_masks.append(gen_mask)
                    graphs.append(graph)
                self.syn_houses = houses
                self.syn_door_masks = door_masks
                self.syn_self_masks = self_masks
                self.syn_gen_masks = gen_masks
                self.syn_graphs = graphs
                # np.savez_compressed(f'processed_rplan/rplan_{set_name}_{target_set}_syn', graphs=self.syn_graphs, houses=self.syn_houses, boundarys=self.syn_boundarys,
                #         door_masks=self.syn_door_masks, self_masks=self.syn_self_masks, gen_masks=self.syn_gen_masks, bound_masks=self.syn_bound_masks)

    def __len__(self):
        return len(self.houses)

    def __getitem__(self, idx):
        # idx = int(idx//20)
        arr = self.houses[idx][:, :self.num_coords].astype(np.float32)
        graph = np.concatenate((self.graphs[idx], np.zeros([200-len(self.graphs[idx]), 3])), 0)

        # Calculate dynamic indices based on hyperparameters
        rtype_start = self.num_coords
        rtype_end = rtype_start + self.rtype_dim
        corner_start = rtype_end
        corner_end = corner_start + self.corner_index_dim
        room_start = corner_end
        room_end = room_start + self.room_index_dim
        padding_idx = room_end
        connections_start = padding_idx + 1
        
        room_types = np.argmax(self.houses[idx][:, rtype_start:rtype_end], axis=1)
        room_indices = np.argmax(self.houses[idx][:, room_start:room_end], axis=1)
        src_key_padding_mask = 1-self.houses[idx][:, padding_idx]
        gen_mask = self.gen_masks[idx].astype(np.uint8)
        self_mask = self.self_masks[idx].astype(np.uint8)
        door_mask = self.door_masks[idx].astype(np.uint8)
        fixed_mask = self.houses[idx][:, connections_start+2:]
        
        if self.set_name == 'train' and random.random() < 0.5: # Do not consider boundary
            room_end = np.where(room_types == 16)[0][0]
            src_key_padding_mask[room_end:] = 1
            gen_mask.fill(1)
            gen_mask[:room_end, :room_end] = 0
            self_mask[gen_mask == 1] = 1
            door_mask[gen_mask == 1] = 1
        
        if self.set_name == 'train' and random.random() < 0.5: # Fix some rooms
            room_end = np.where(room_types == 16)[0][0]
            unique_room_types = np.unique(room_indices[:room_end])
            k = random.randint(1, len(unique_room_types)-1)
            freeze_room = random.sample(list(unique_room_types), k)
            for room_instance in freeze_room:
                src_key_padding_mask[room_indices == room_instance] = 1
                fixed_mask[room_indices == room_instance] = 1
                
        cond = {
                'door_mask': door_mask,
                'self_mask': self_mask,
                'gen_mask': gen_mask,
                'room_types': self.houses[idx][:, rtype_start:rtype_end],
                'corner_indices': self.houses[idx][:, corner_start:corner_end].astype(np.uint8),
                'room_indices': self.houses[idx][:, room_start:room_end].astype(np.uint8),
                'src_key_padding_mask': src_key_padding_mask,
                'connections': self.houses[idx][:, connections_start:connections_start+2].astype(np.uint8),
                'fixed_mask': fixed_mask,
                'graph': graph,
                }
    
        if self.set_name == 'eval':
            syn_graph = np.concatenate((self.syn_graphs[idx], np.zeros([200-len(self.syn_graphs[idx]), 3])), 0)
            assert (graph == syn_graph).all(), idx
            
            room_types = np.argmax(self.syn_houses[idx][:, rtype_start:rtype_end], axis=1)
            syn_src_key_padding_mask = 1-self.syn_houses[idx][:, padding_idx]
            syn_gen_mask = self.syn_gen_masks[idx].astype(np.uint8)
            syn_self_mask = self.syn_self_masks[idx].astype(np.uint8)
            syn_door_mask = self.syn_door_masks[idx].astype(np.uint8)
            syn_fixed_mask = self.syn_houses[idx][:, connections_start+2:]
            
            # # Do not consider boundary
            # room_end = np.where(room_types == 16)[0][0]
            # syn_src_key_padding_mask[room_end:] = 1
            # syn_gen_mask.fill(1)
            # syn_gen_mask[:room_end, :room_end] = 0
            # syn_self_mask[syn_gen_mask == 1] = 1
            # syn_door_mask[syn_gen_mask == 1] = 1
            # syn_room_indices = self.syn_houses[idx][:, room_start:room_start+self.room_index_dim].astype(np.uint8)
            # syn_room_indices = np.argmax(syn_room_indices, axis=-1)
            # graph_filter = [syn_room_indices.max()-1, syn_room_indices.max() - 2]
            # syn_graph[np.isin(syn_graph[:, 0], graph_filter)] = 0
            # syn_graph[np.isin(syn_graph[:, -1], graph_filter)] = 0
            
            cond.update({
                'syn_door_mask': syn_door_mask,
                'syn_self_mask': syn_self_mask,
                'syn_gen_mask': syn_gen_mask,
                'syn_room_types': self.syn_houses[idx][:, rtype_start:rtype_end],
                'syn_corner_indices': self.syn_houses[idx][:, corner_start:corner_end],
                'syn_room_indices': self.syn_houses[idx][:, room_start:room_start+self.room_index_dim],
                'syn_src_key_padding_mask': syn_src_key_padding_mask,
                'syn_connections': self.syn_houses[idx][:, connections_start:connections_start+2],
                'syn_fixed_mask': syn_fixed_mask,
                'syn_graph': syn_graph,
                'graph': syn_graph,
                })
            arr_syn = self.syn_houses[idx][:, :self.num_coords]
        if self.set_name == 'train':
            #### Random Rotate
            rotation = random.randint(0,3)
            if rotation == 1:
                arr[:, [0, 1]] = arr[:, [1, 0]]
                arr[:, 0] = -arr[:, 0]
            elif rotation == 2:
                arr[:, [0, 1]] = -arr[:, [1, 0]]
            elif rotation == 3:
                arr[:, [0, 1]] = arr[:, [1, 0]]
                arr[:, 1] = -arr[:, 1]

            ## To generate any rotation uncomment this

            # if self.non_manhattan:
                # theta = random.random()*np.pi/2
                # rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                             # [np.sin(theta), np.cos(theta), 0]])
                # arr = np.matmul(arr,rot_mat)[:,:2]

            # Random Scale
            # arr = arr * np.random.normal(1., .5)

            # Random Shift
            # arr[:, 0] = arr[:, 0] + np.random.normal(0., .1)
            # arr[:, 1] = arr[:, 1] + np.random.normal(0., .1)
        if not self.analog_bit:
            if self.set_name == 'train':
                arr = np.transpose(arr, [1, 0])
                return arr.astype(np.float32), cond
            else:
                arr = np.transpose(arr, [1, 0])
                arr_syn = np.transpose(arr_syn, [1, 0])
                return arr.astype(np.float32), arr_syn.astype(np.float32), cond
        else:
            ONE_HOT_RES = 256
            arr_onehot = np.zeros((ONE_HOT_RES*2, arr.shape[1])) - 1
            xs = ((arr[:, 0]+1)*(ONE_HOT_RES/2)).astype(int)
            ys = ((arr[:, 1]+1)*(ONE_HOT_RES/2)).astype(int)
            xs = np.array([get_bin(x, 8) for x in xs])
            ys = np.array([get_bin(x, 8) for x in ys])
            arr_onehot = np.concatenate([xs, ys], 1)
            arr_onehot = np.transpose(arr_onehot, [1, 0])
            arr_onehot[arr_onehot==0] = -1
            return arr_onehot.astype(float), cond

    def make_sequence(self, edges):
        polys = []
        v_curr = tuple(edges[0][:2])
        e_ind_curr = 0
        e_visited = [0]
        seq_tracker = [v_curr]
        find_next = False
        while len(e_visited) < len(edges):
            if find_next == False:
                if v_curr == tuple(edges[e_ind_curr][2:]):
                    v_curr = tuple(edges[e_ind_curr][:2])
                else:
                    v_curr = tuple(edges[e_ind_curr][2:])
                find_next = not find_next 
            else:
                # look for next edge
                for k, e in enumerate(edges):
                    if k not in e_visited:
                        if (v_curr == tuple(e[:2])):
                            v_curr = tuple(e[2:])
                            e_ind_curr = k
                            e_visited.append(k)
                            break
                        elif (v_curr == tuple(e[2:])):
                            v_curr = tuple(e[:2])
                            e_ind_curr = k
                            e_visited.append(k)
                            break

            # extract next sequence
            if v_curr == seq_tracker[-1]:
                polys.append(seq_tracker)
                for k, e in enumerate(edges):
                    if k not in e_visited:
                        v_curr = tuple(edges[0][:2])
                        seq_tracker = [v_curr]
                        find_next = False
                        e_ind_curr = k
                        e_visited.append(k)
                        break
            else:
                seq_tracker.append(v_curr)
        polys.append(seq_tracker)

        return polys

    def build_graph(self, rms_type, r_boundarys, graph):
        # create edges
        triples = []
        nodes = rms_type
        # encode connections
        for k in range(len(nodes)):
            for l in range(len(nodes)):
                if l > k:
                    is_adjacent = any([True for e_map in graph if (l in e_map) and (k in e_map)])
                    if is_adjacent:
                        if 'train' in self.set_name:
                            triples.append([k, 1, l])
                        else:
                            triples.append([k, 1, l])
                    else:
                        if 'train' in self.set_name:
                            triples.append([k, -1, l])
                        else:
                            triples.append([k, -1, l])
                            
        # rms_masks = []
        # im_size = 256
        # fp_mk = np.zeros((out_size, out_size))
        # for k in range(len(nodes)):
        #     if nodes[k] == 17 or nodes[k] == 15:
        #         rms_masks.append(r_boundarys[k])
        #         continue
        #     # add rooms and doors
        #     eds = r_boundarys[k]
        #     # draw rooms
        #     rm_im = Image.new('L', (im_size, im_size))
        #     dr = ImageDraw.Draw(rm_im)
        #     for eds_poly in [eds]:
        #         poly = eds_poly + [eds_poly[0]]
        #         poly = [tuple(item) for item in poly]
        #         if len(poly) >= 2:
        #             dr.polygon(poly, fill='white')
        #         else:
        #             print("Empty room")
        #             exit(0)
        #     rm_im = rm_im.resize((out_size, out_size))
        #     rm_arr = np.array(rm_im)
        #     inds = np.where(rm_arr>0)
        #     rm_arr[inds] = 1.0
        #     rms_masks.append(rm_arr)
        #     if rms_type[k] != 15 and rms_type[k] != 17 and rms_type[k] != 18:
        #         fp_mk[inds] = k+1
        # # trick to remove overlap
        # for k in range(len(nodes)):
        #     if rms_type[k] != 15 and rms_type[k] != 17 and rms_type[k] != 18:
        #         rm_arr = np.zeros((out_size, out_size))
        #         inds = np.where(fp_mk==k+1)
        #         rm_arr[inds] = 1.0
        #         rms_masks[k] = rm_arr

        # convert to array
        nodes = np.array(nodes)
        triples = np.array(triples)
        return nodes, triples


def is_adjacent(box_a, box_b, threshold=0.03):
    x0, y0, x1, y1 = box_a
    x2, y2, x3, y3 = box_b
    h1, h2 = x1-x0, x3-x2
    w1, w2 = y1-y0, y3-y2
    xc1, xc2 = (x0+x1)/2.0, (x2+x3)/2.0
    yc1, yc2 = (y0+y1)/2.0, (y2+y3)/2.0
    delta_x = np.abs(xc2-xc1) - (h1 + h2)/2.0
    delta_y = np.abs(yc2-yc1) - (w1 + w2)/2.0
    delta = max(delta_x, delta_y)
    return delta < threshold

def polygon_to_edges(r_boundary):
    edges = []
    num_points = len(r_boundary)
    for i in range(num_points):
        x1, y1 = r_boundary[i]
        x2, y2 = r_boundary[(i + 1) % num_points]  # Use modulo operation to close polygon
        edges.append([x1, y1, x2, y2, 0, 0])
    return edges

def reader(filename,):
    with open(filename) as f:
        info =json.load(f)
    
    rms_type=info['room_types']
    r_boundarys=info['room_boundaries']
    # visualize_polygon(r_boundarys)
       
    graph=info['graph']
  
    rms_type = np.array(rms_type)
        
    return rms_type, graph, r_boundarys

if __name__ == '__main__':
    dataset = RPlanhgDataset('eval', False, 8)
