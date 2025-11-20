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


def load_msd_data(
    batch_size,
    analog_bit,
    target_set,
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
    print(f"loading {set_name}")
    deterministic = False if set_name=='train' else True
    dataset = MSDDataset(set_name, analog_bit, target_set, non_manhattan=False,
                         rtype_dim=rtype_dim,
                         corner_index_dim=corner_index_dim,
                         room_index_dim=room_index_dim,
                         max_num_points=max_num_points)
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=5, drop_last=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=5, drop_last=False
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


get_bin = lambda x, z: [int(y) for y in format(x, 'b').zfill(z)]
get_one_hot = lambda x, z: np.eye(z)[x]

fp_num_list = [[15,16,17], [18,19,20], [21,22,23,24], [25,26,27,28,29,30,31]]

class MSDDataset(Dataset):
    def __init__(self, set_name, analog_bit, target_set, non_manhattan=False, 
                 rtype_dim=25, corner_index_dim=32, room_index_dim=32,
                 max_num_points=200):
        super().__init__()
        base_dir = './datasets/msd'
        self.non_manhattan = non_manhattan
        self.target_set = target_set
        self.set_name = set_name
        self.analog_bit = analog_bit
        
        # Hyperparameters
        self.num_coords = 2
        self.rtype_dim = rtype_dim
        self.corner_index_dim = corner_index_dim  # also defines max corners per room
        self.room_index_dim = room_index_dim      # also defines max rooms per house
        self.max_num_points = max_num_points
        
        self.subgraphs = []
        self.org_graphs = []
        self.org_houses = []
        max_num_points = self.max_num_points
        if self.set_name == 'eval':
            cnumber_dist = np.load(f'processed_msd/msd_train_cndist.npz', allow_pickle=True)['cnumber_dist'].item()
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
            fp_size_list = []
            for line in tqdm(lines):
                cnt=cnt+1
                file_name = f'{base_dir}/{line[:-1]}'
                rms_type,graph,r_boundarys,h_boundarys=reader(file_name)
                
                fp_size = len([x for x in rms_type if x != 15 and x != 17])
                graph_nodes, graph_edges = self.build_graph(rms_type, r_boundarys, graph)

                skip_file = False
                room_boundaries = []
                for room_mask in r_boundarys:

                    if len(room_mask) > self.corner_index_dim:  # Skip entire file if overly long contour is found
                        skip_file = True
                        break
                    room_boundaries.append(room_mask)

                if len(r_boundarys) >= self.room_index_dim:  # Skip entire file if too many rooms found
                    skip_file = True
                
                if skip_file:
                    continue  # Skip current file directly

                if self.set_name == 'train' and fp_size in fp_num_list[self.target_set]:
                    continue
                if self.set_name == 'eval' and fp_size not in fp_num_list[self.target_set]:
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
                    room[0] = np.reshape(room[0], [len(room[0]), 2])/512. - 0.5 # [[x0,y0],[x1,y1],...,[x15,y15]] and map to 0-1 - > -0.5, 0.5
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
                    else:
                        padding_mask = np.repeat(1, num_room_corners)
                        padding_mask = np.expand_dims(padding_mask, 1)
                    # Generating corner bounds for attention masks
                    connections = np.array([[i,(i+1)%num_room_corners] for i in range(num_room_corners)])
                    connections += num_points
                    corner_bounds.append([num_points, num_points+num_room_corners])
                    num_points += num_room_corners
                    room = np.concatenate((room[0], rtype, corner_index, room_index, padding_mask, connections), 1)
                    house.append(room)

                house_layouts = np.concatenate(house, 0)
                if len(house_layouts)>max_num_points:
                    continue
                # Calculate feature dimension: coords + rtype + corner_index + room_index + padding_mask + connections
                feature_dim = self.num_coords + self.rtype_dim + self.corner_index_dim + self.room_index_dim + 1 + 2
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
                os.makedirs('processed_msd', exist_ok=True)
                np.savez_compressed(f'processed_msd/msd_{set_name}_cndist', cnumber_dist=cnumber_dist)

            if set_name=='eval':
                houses = []
                graphs = []
                door_masks = []
                self_masks = []
                gen_masks = []
                len_house_layouts = 0
                for index, (h, graph) in tqdm(enumerate(zip(self.org_houses, self.org_graphs)), desc='processing syn dataset', total=len(self.org_houses)):
                    if np.sum([len(room[0]) for room in h])>max_num_points:
                        continue
                    house = []
                    corner_bounds = []
                    num_points = 0
                    # num_room_corners_total = [len(room[0]) for room in h]
                    num_room_corners_total = [cnumber_dist[room[1]][random.randint(0, len(cnumber_dist[room[1]])-1)] for room in h]
                    while np.sum(num_room_corners_total)>=max_num_points:
                        num_room_corners_total = [cnumber_dist[room[1]][random.randint(0, len(cnumber_dist[room[1]])-1)] for room in h]
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
                        else:
                            padding_mask = np.repeat(1, num_room_corners)
                            padding_mask = np.expand_dims(padding_mask, 1)
                        # Generating corner bounds for attention masks
                        connections = np.array([[i,(i+1)%num_room_corners] for i in range(num_room_corners)])
                        connections += num_points
                        corner_bounds.append([num_points, num_points+num_room_corners])
                        num_points += num_room_corners
                        if room[1] not in [15, 18]:
                            room = np.concatenate((np.zeros([num_room_corners, 2]), rtype, corner_index, room_index, padding_mask, connections), 1)
                        else:
                            room = np.concatenate((room[0], rtype, corner_index, room_index, padding_mask, connections), 1)
                        house.append(room)

                    house_layouts = np.concatenate(house, 0)
                        
                    # Calculate feature dimension: coords + rtype + corner_index + room_index + padding_mask + connections
                    feature_dim = self.num_coords + self.rtype_dim + self.corner_index_dim + self.room_index_dim + 1 + 2
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
        graph = np.concatenate((self.graphs[idx], np.zeros([500-len(self.graphs[idx]), 3])), 0)

        # Calculate dynamic indices based on hyperparameters
        rtype_start = self.num_coords
        rtype_end = rtype_start + self.rtype_dim
        corner_start = rtype_end
        corner_end = corner_start + self.corner_index_dim
        room_start = corner_end
        room_end = room_start + self.room_index_dim
        padding_idx = room_end
        connections_start = padding_idx + 1
        connections_end = connections_start + 2
        
        room_types = np.argmax(self.houses[idx][:, rtype_start:rtype_end], axis=1)
        room_indices = np.argmax(self.houses[idx][:, room_start:room_end], axis=1)
        src_key_padding_mask = 1-self.houses[idx][:, padding_idx]
        gen_mask = self.gen_masks[idx].astype(np.uint8)
        self_mask = self.self_masks[idx].astype(np.uint8)
        door_mask = self.door_masks[idx].astype(np.uint8)
        
        # if self.set_name == 'train' and random.random() < 0.5: # Fix some rooms
        #     unique_room_types = np.unique(room_indices)
        #     unique_room_types = unique_room_types[unique_room_types != 0]
        #     k = random.randint(1, len(unique_room_types)-1)
        #     freeze_room = random.sample(list(unique_room_types), k)
        #     for room_instance in freeze_room:
        #         src_key_padding_mask[room_indices == room_instance] = 1
                
        cond = {
                'door_mask': door_mask,
                'self_mask': self_mask,
                'gen_mask': gen_mask,
                'room_types': self.houses[idx][:, rtype_start:rtype_end],
                'corner_indices': self.houses[idx][:, corner_start:corner_end].astype(np.uint8),
                'room_indices': self.houses[idx][:, room_start:room_end].astype(np.uint8),
                'src_key_padding_mask': src_key_padding_mask,
                'connections': self.houses[idx][:, connections_start:connections_end].astype(np.uint8),
                'graph': graph,
                }
    
        if self.set_name == 'eval':
            syn_graph = np.concatenate((self.syn_graphs[idx], np.zeros([500-len(self.syn_graphs[idx]), 3])), 0)
            assert (graph == syn_graph).all(), idx
            
            room_types = np.argmax(self.syn_houses[idx][:, rtype_start:rtype_end], axis=1)
            syn_src_key_padding_mask = 1-self.syn_houses[idx][:, padding_idx]
            syn_gen_mask = self.syn_gen_masks[idx].astype(np.uint8)
            syn_self_mask = self.syn_self_masks[idx].astype(np.uint8)
            syn_door_mask = self.syn_door_masks[idx].astype(np.uint8)
            
            cond.update({
                'syn_door_mask': syn_door_mask,
                'syn_self_mask': syn_self_mask,
                'syn_gen_mask': syn_gen_mask,
                'syn_room_types': self.syn_houses[idx][:, rtype_start:rtype_end],
                'syn_corner_indices': self.syn_houses[idx][:, corner_start:corner_end],
                'syn_room_indices': self.syn_houses[idx][:, room_start:room_end],
                'syn_src_key_padding_mask': syn_src_key_padding_mask,
                'syn_connections': self.syn_houses[idx][:, connections_start:connections_end],
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

        # convert to array
        nodes = np.array(nodes)
        triples = np.array(triples)
        return nodes, triples


def reader(filename,):
    with open(filename) as f:
        info =json.load(f)
    
    rms_type=info['room_types']
    r_boundarys=info['room_boundaries']
    h_boundarys=info['bondary']
    # visualize_polygon(r_boundarys)
       
    graph=info['graph']
  
    rms_type = np.array(rms_type)
        
    return rms_type, graph, r_boundarys, h_boundarys


if __name__ == '__main__':
    dataset = MSDDataset('eval', False, 8)

