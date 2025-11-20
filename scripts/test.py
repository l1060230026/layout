"""
Generate select layout
"""

import argparse
import os

import numpy as np
import math
import torch as th
import cv2 as cv

import io
from PIL import Image, ImageDraw
import drawSvg as drawsvg
import cairosvg
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_fid.fid_score import calculate_fid_given_paths
from ConDiffPlan.rplanhg_datasets import load_rplanhg_data, reader, get_one_hot
from ConDiffPlan import dist_util, logger
from ConDiffPlan.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    update_arg_parser,
)
import webcolors
import networkx as nx
from collections import defaultdict
from shapely.geometry import Polygon
from shapely.geometry.base import geom_factory
from shapely.geos import lgeos
from shapely.ops import unary_union

def get_graph(indx, g_true, ID_COLOR, draw_graph, save_svg):
    # build true graph
    G_true = nx.Graph()
    colors_H = []
    node_size = []
    edge_color = []
    linewidths = []
    edgecolors = []
    # add nodes
    for k, label in enumerate(g_true[0]):
        _type = label
        if _type >= 0 and _type not in [16, 18, 19]:
            G_true.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
            node_size.append(1000)
            edgecolors.append('blue')
            linewidths.append(0.0)
    # add outside node
    # G_true.add_nodes_from([(-1, {'label':-1})])
    # colors_H.append("white")
    # node_size.append(750)
    # edgecolors.append('black')
    # linewidths.append(3.0)
    # add edges
    for k, m, l in g_true[1]:
        k = int(k)
        l = int(l)
        _type_k = g_true[0][k]
        _type_l = g_true[0][l]
        if m > 0 and (_type_k not in [16, 18, 19] and _type_l not in [16, 18, 19]):
            G_true.add_edges_from([(k, l)])
            edge_color.append('#D3A2C7')
        # elif m > 0 and (_type_k==16 or _type_l==16):
        #     if _type_k==16:
        #         G_true.add_edges_from([(l, -1)])
        #     else:
        #         G_true.add_edges_from([(k, -1)])
        #     edge_color.append('#727171')
    if draw_graph:
        plt.figure()
        pos = nx.nx_agraph.graphviz_layout(G_true, prog='neato')
        nx.draw(G_true, pos, node_size=node_size, linewidths=linewidths, node_color=colors_H, font_size=14, font_color='white',\
                font_weight='bold', edgecolors=edgecolors, width=4.0, with_labels=False)
        if save_svg:
            plt.savefig(f'outputs/graphs_gt/{indx}.svg')
        else:
            plt.savefig(f'outputs/graphs_gt/{indx}.jpg')
        plt.close('all')
    return G_true

def estimate_graph(indx, polys, nodes, G_gt, ID_COLOR, draw_graph, save_svg):
    nodes = np.array(nodes)
    
    G_gt = get_graph(indx, [nodes, G_gt], ID_COLOR, draw_graph, save_svg)
    
    G_estimated = nx.Graph()
    colors_H = []
    node_size = []
    edge_color = []
    linewidths = []
    edgecolors = []
    edge_labels = {}
    # add nodes
    for k, label in enumerate(nodes):
        _type = label
        if _type >= 0 and _type not in [16, 18, 19]:
            G_estimated.add_nodes_from([(k, {'label':k})])
            colors_H.append(ID_COLOR[_type])
            node_size.append(1000)
            linewidths.append(0.0)
    # # add outside node
    # G_estimated.add_nodes_from([(-1, {'label':-1})])
    # colors_H.append("white")
    # node_size.append(750)
    # edgecolors.append('black')
    # linewidths.append(3.0)
    # add node-to-door connections
    doors_inds = np.where((nodes == 16) | (nodes == 18) | (nodes == 19))[0]
    rooms_inds = np.where((nodes != 16) & (nodes != 18) & (nodes != 19))[0]
    doors_rooms_map = defaultdict(list)
    for k in doors_inds:
        for l in rooms_inds:
            if k > l:
                p1, p2 = polys[k], polys[l]
                p1, p2 = Polygon(p1), Polygon(p2)
                if not p1.is_valid:
                    p1 = geom_factory(lgeos.GEOSMakeValid(p1._geom))
                if not p2.is_valid:
                    p2 = geom_factory(lgeos.GEOSMakeValid(p2._geom))
                iou = p1.intersection(p2).area/ p1.union(p2).area
                if iou > 0 and iou < 0.2:
                    doors_rooms_map[k].append((l, iou))
    # draw connections
    for k in doors_rooms_map.keys():
        _conn = doors_rooms_map[k]
        _conn = sorted(_conn, key=lambda tup: tup[1], reverse=True)
        _conn_top2 = _conn[:2]
        if nodes[k] != 11:
            if len(_conn_top2) > 1:
                l1, l2 = _conn_top2[0][0], _conn_top2[1][0]
                edge_labels[(l1, l2)] = k
                G_estimated.add_edges_from([(l1, l2)])
        else:
            if len(_conn) > 0:
                l1 = _conn[0][0]
                edge_labels[(-1, l1)] = k
                G_estimated.add_edges_from([(-1, l1)])
    # add missed edges
    G_estimated_complete = G_estimated.copy()
    for k, l in G_gt.edges():
        if not G_estimated.has_edge(k, l):
            G_estimated_complete.add_edges_from([(k, l)])
    # add edges colors
    colors = []
    mistakes = 0
    for k, l in G_estimated_complete.edges():
        if G_gt.has_edge(k, l) and not G_estimated.has_edge(k, l):
            colors.append('yellow')
            mistakes += 1
        elif G_estimated.has_edge(k, l) and not G_gt.has_edge(k, l):
            colors.append('red')
            mistakes += 1
        elif G_estimated.has_edge(k, l) and G_gt.has_edge(k, l):
            colors.append('green')
        else:
            print('ERR')
    if draw_graph:
        plt.figure()
        pos = nx.nx_agraph.graphviz_layout(G_estimated_complete, prog='neato')
        weights = [4 for u, v in G_estimated_complete.edges()]
        nx.draw(G_estimated_complete, pos, edge_color=colors, linewidths=linewidths, edgecolors=edgecolors, node_size=node_size, node_color=colors_H, font_size=14, font_weight='bold', font_color='white', width=weights, with_labels=False)
        if save_svg:
            plt.savefig(f'outputs/graphs_pred/{indx}.svg')
        else:
            plt.savefig(f'outputs/graphs_pred/{indx}.jpg')
        plt.close('all')
    return mistakes

def build_graph(rms_type, r_boundarys, graph):
    # create edges
    triples = []
    nodes = rms_type
    # encode connections
    for k in range(len(nodes)):
        for l in range(len(nodes)):
            if l > k:
                is_adjacent = any([True for e_map in graph if (l in e_map) and (k in e_map)])
                if is_adjacent:
                    triples.append([k, 1, l])
                else:
                    triples.append([k, -1, l])
                        
    nodes = np.array(nodes)
    triples = np.array(triples)
    return nodes, triples

def make_sequence(edges):
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


def calculate_occupancy_ratio(polys, types, door_indices, boundary_type=19):
    """
    Calculate the ratio of room areas within the house boundary to the total boundary area.
    This function properly handles overlapping rooms by using unary_union.
    
    Args:
        polys: List of polygons (each polygon is a list of (x, y) coordinates)
        types: List of room types corresponding to each polygon
        door_indices: List of room type indices that represent doors (e.g., [16, 18, 19])
        boundary_type: Room type that represents the house boundary (default: 19)
    
    Returns:
        occupancy_ratio: The ratio of room area to boundary area (0 to 1)
    """
    # Find the house boundary polygon
    boundary_poly = None
    room_polys = []
    
    for poly, room_type in zip(polys, types):
        if len(poly) < 3:  # Skip invalid polygons
            continue
            
        # Find boundary (type 19)
        if room_type == boundary_type:
            try:
                boundary_shapely = Polygon(poly)
                if not boundary_shapely.is_valid:
                    boundary_shapely = geom_factory(lgeos.GEOSMakeValid(boundary_shapely._geom))
                if boundary_poly is None:
                    boundary_poly = boundary_shapely
                else:
                    # If multiple boundaries, take the union
                    boundary_poly = boundary_poly.union(boundary_shapely)
            except Exception as e:
                print(f"Warning: Failed to create boundary polygon: {e}")
                continue
        
        # Collect room polygons (excluding doors, boundaries, and empty type)
        elif room_type not in door_indices and room_type != 0 and room_type != boundary_type:
            try:
                room_shapely = Polygon(poly)
                if not room_shapely.is_valid:
                    room_shapely = geom_factory(lgeos.GEOSMakeValid(room_shapely._geom))
                room_polys.append(room_shapely)
            except Exception as e:
                print(f"Warning: Failed to create room polygon: {e}")
                continue
    
    # If no boundary found, return 0
    if boundary_poly is None or not boundary_poly.is_valid:
        print("Warning: No valid boundary found")
        return 0.0
    
    # Calculate boundary area
    boundary_area = boundary_poly.area
    if boundary_area == 0:
        return 0.0
    
    # If no rooms found, return 0
    if len(room_polys) == 0:
        print("Warning: No valid rooms found")
        return 0.0
    
    # Use unary_union to merge all rooms and handle overlaps correctly
    try:
        # First, clip all rooms to the boundary
        clipped_rooms = []
        for room_poly in room_polys:
            intersection = room_poly.intersection(boundary_poly)
            if not intersection.is_empty:
                clipped_rooms.append(intersection)
        
        if len(clipped_rooms) == 0:
            return 0.0
        
        # Union all clipped rooms to get total coverage (avoiding double-counting overlaps)
        unified_rooms = unary_union(clipped_rooms)
        total_room_area = unified_rooms.area
        
    except Exception as e:
        print(f"Warning: Failed to calculate union: {e}")
        return 0.0
    
    # Calculate occupancy ratio
    occupancy_ratio = total_room_area / boundary_area
    
    return occupancy_ratio


def save_samples(
        sample, ext, model_kwargs, 
        tmp_count, num_room_types, 
        save_gif=False, save_edges=False,
        door_indices = [16, 19, 18], ID_COLOR=None,
        is_syn=False, draw_graph=False, save_svg=False,
        room_class=None, show_room_names=False):
    prefix = 'syn_' if is_syn else ''
    graph_errors = []
    occupancy_ratios = []  # Store occupancy ratios for each sample
    if room_class is not None:
        reversed_room_class = {value: key for key, value in room_class.items()}
    if not save_gif:
        sample = sample[-1:]
    for i in tqdm(range(sample.shape[1])):
        resolution = 256
        images = []
        images2 = []
        images3 = []
        for k in range(sample.shape[0]):
            draw = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))
            draw2 = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw2.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))
            draw3 = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw3.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='black'))
            draw_color = drawsvg.Drawing(resolution, resolution, displayInline=False)
            draw_color.append(drawsvg.Rectangle(0,0,resolution,resolution, fill='white'))
            polys = []
            types = []
            indices = []
            for j, point in (enumerate(sample[k][i])):
                if model_kwargs[f'{prefix}gen_mask'][i][0][j]==1:
                    continue
                point = point.cpu().data.numpy()
                if j==0:
                    poly = []
                if j>0 and (model_kwargs[f'{prefix}room_indices'][i, j]!=model_kwargs[f'{prefix}room_indices'][i, j-1]).any():
                    polys.append(poly)
                    types.append(c)
                    indices.append(d)
                    poly = []
                pred_center = False
                if pred_center:
                    point = point/2 + 1
                    point = point * resolution//2
                else:
                    point = point/2 + 0.5
                    point = point * resolution
                poly.append((point[0], point[1]))
                c = np.argmax(model_kwargs[f'{prefix}room_types'][i][j-1].cpu().numpy())
                d = np.argmax(model_kwargs[f'{prefix}room_indices'][i][j-1].cpu().numpy())
            polys.append(poly)
            types.append(c)
            indices.append(d)
            
            description = ""
            for index, c in zip(indices, types):
                if c in reversed_room_class:
                    description += f'{index}: {reversed_room_class[c]}, '
            print(description)
            
            for poly, c, d in zip(polys, types, indices):
                if c in door_indices or c==0:
                    continue
                room_type = c
                c = webcolors.hex_to_rgb(ID_COLOR[c])
                
                # Calculate the center point of the room polygon as text position
                points = np.array(poly)
                center_x = np.mean(points[:, 0])
                center_y = np.mean(points[:, 1])
                
                # Get room name and add text
                room_name = None
                if show_room_names:
                    room_name = f'{d}'
                
                draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='black', stroke_width=1))
                if room_name:
                    draw_color.append(drawsvg.Text(room_name, 12, center_x, center_y, text_anchor='middle', dominant_baseline='middle', fill='black'))
                
                draw.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill='black', fill_opacity=0.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                draw2.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                for corner in poly:
                    draw.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
                    draw3.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
            for poly, c in zip(polys, types):
                if c not in door_indices:
                    continue
                room_type = c
                c = webcolors.hex_to_rgb(ID_COLOR[c])
                # poly = line_to_rectangle(poly[0], poly[1], 6)
                # draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='black', stroke_width=1))
                if room_type == 19:
                    draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=0.0, stroke='black', stroke_width=1))
                else:
                    draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='black', stroke_width=1))
                draw.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill='black', fill_opacity=0.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                draw2.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                for corner in poly:
                    draw.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
                    draw3.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
            
            # Calculate occupancy ratio (room coverage within house boundary)
            if k == sample.shape[0] - 1:
                occupancy_ratio = calculate_occupancy_ratio(polys, types, door_indices)
                occupancy_ratios.append(occupancy_ratio)
                print(f"Sample {tmp_count+i}: Occupancy Ratio = {occupancy_ratio:.4f} (Gap Ratio = {1-occupancy_ratio:.4f})")
            
            images.append(Image.open(io.BytesIO(cairosvg.svg2png(draw.asSvg()))))
            images2.append(Image.open(io.BytesIO(cairosvg.svg2png(draw2.asSvg()))))
            images3.append(Image.open(io.BytesIO(cairosvg.svg2png(draw3.asSvg()))))
            if k==sample.shape[0]-1 or True:
                if save_edges:
                    draw.saveSvg(f'outputs/{ext}/{tmp_count+i}_{k}_{ext}.svg')
                if save_svg:
                    draw_color.saveSvg(f'outputs/{ext}/{tmp_count+i}c_{k}_{ext}.svg')
                else:
                    Image.open(io.BytesIO(cairosvg.svg2png(draw_color.asSvg()))).save(f'outputs/{ext}/{tmp_count+i}c_{ext}.png')
            if k==sample.shape[0]-1:
                if 'graph' in model_kwargs:
                    graph_errors.append(estimate_graph(tmp_count+i, polys, types, model_kwargs[f'{prefix}graph'][i], ID_COLOR=ID_COLOR, draw_graph=draw_graph, save_svg=save_svg))
                else:
                    graph_errors.append(0)
        if save_gif:
            imageio.mimwrite(f'outputs/gif/{tmp_count+i}.gif', images, fps=10, loop=1)
            imageio.mimwrite(f'outputs/gif/{tmp_count+i}_v2.gif', images2, fps=10, loop=1)
            imageio.mimwrite(f'outputs/gif/{tmp_count+i}_v3.gif', images3, fps=10, loop=1)
    
    # Print average occupancy ratio
    if len(occupancy_ratios) > 0:
        avg_occupancy = np.mean(occupancy_ratios)
        avg_gap = 1 - avg_occupancy
        print(f"\nAverage Occupancy Ratio: {avg_occupancy:.4f}")
        print(f"Average Gap Ratio: {avg_gap:.4f}")
    
    return graph_errors, occupancy_ratios

class RPlanhgDataset():
    def __init__(self, length=None, image_id=None, non_manhattan=False):
        base_dir = './datasets/rplan'
        max_num_points = 100
        self.max_num_points=max_num_points
        self.num_coords=2
        subgraphs = []
        self.org_graphs = []
        self.org_houses = []
        with open(f'{base_dir}/list.txt') as f:
            lines = f.readlines()
        if length is not None:
            file_names = lines[:length]
        else:
            file_names = lines
        for name in file_names:
            file_name = f'{base_dir}/{name[:-1]}'
            if image_id is not None:
                file_name = f'{base_dir}/{image_id}.json'
            
            rms_type, graph, r_boundarys=reader(file_name)
            
            fp_size = len([x for x in rms_type if x != 15 and x != 17 and x != 18])
            graph_nodes, graph_edges = build_graph(rms_type, r_boundarys, graph)

            skip_file = False
            room_boundaries = []
            for room_mask in r_boundarys:

                if len(room_mask) > 32:  # Skip entire file if overly long contour is found
                    skip_file = True
                    break
                room_boundaries.append(room_mask)
            
            if skip_file:
                continue  # Skip current file directly
                
            if fp_size not in [5, 6, 8]:
                continue
            a = [rms_type, room_boundaries,]
            subgraphs.append(a)
            
            house = []
            for index, room_type in enumerate(rms_type):
                contours = room_boundaries[index]
                house.append([contours, room_type])
            self.org_graphs.append(graph_edges)
            self.org_houses.append(house)
            
            if image_id is not None:
                break
        
        self.houses = []
        self.door_masks = []
        self.self_masks = []
        self.gen_masks = []
        self.graphs = []
        
        for index, (h, graph) in tqdm(enumerate(zip(self.org_houses, self.org_graphs)), total=len(self.org_houses), desc='processing dataset'):
            house = []
            corner_bounds = []
            num_points = 0
            for i, room in enumerate(h):
                # if room[1]>10:
                #     room[1] = {15:11, 17:12, 16:13}[room[1]]
                room[0] = np.reshape(room[0], [len(room[0]), 2])/256. - 0.5 # [[x0,y0],[x1,y1],...,[x15,y15]] and map to 0-1 - > -0.5, 0.5
                room[0] = room[0] * 2 # map to [-1, 1]
                # Adding conditions
                num_room_corners = len(room[0])
                rtype = np.repeat(np.array([get_one_hot(room[1]+1, 25)]), num_room_corners, 0)
                room_index = np.repeat(np.array([get_one_hot(len(house)+1, 32)]), num_room_corners, 0)
                corner_index = np.array([get_one_hot(x, 32) for x in range(num_room_corners)])
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
            padding = np.zeros((max_num_points-len(house_layouts), 95))
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
            self.houses.append(house_layouts)
            self.door_masks.append(door_mask)
            self.self_masks.append(self_mask)
            self.gen_masks.append(gen_mask)
            self.graphs.append(graph)
            
    def get_data(self, idx):
        arr = th.FloatTensor(self.houses[idx][:, :self.num_coords].astype(np.float32))
        graph = np.concatenate((self.graphs[idx], np.zeros([200-len(self.graphs[idx]), 3])), 0)

        room_types = np.argmax(self.houses[idx][:, self.num_coords:self.num_coords+25], axis=1)
        src_key_padding_mask = 1-self.houses[idx][:, self.num_coords+89]
        gen_mask = self.gen_masks[idx].astype(np.uint8)
        self_mask = self.self_masks[idx].astype(np.uint8)
        door_mask = self.door_masks[idx].astype(np.uint8)

        
        # # Do not consider boundary
        # room_end = np.where(room_types == 16)[0][0]
        # src_key_padding_mask[room_end:] = 1
        # gen_mask.fill(1)
        # gen_mask[:room_end, :room_end] = 0
        # self_mask[gen_mask == 1] = 1
        # door_mask[gen_mask == 1] = 1
        # room_indices = self.houses[idx][:, self.num_coords+57:self.num_coords+89].astype(np.uint8)
        # room_indices = np.argmax(room_indices, axis=-1)
        # graph_filter = [room_indices.max()-1, room_indices.max() - 2]
        # graph[np.isin(graph[:, 0], graph_filter)] = 0
        # graph[np.isin(graph[:, -1], graph_filter)] = 0
        
        cond = {
                'syn_door_mask': th.Tensor(door_mask),
                'syn_self_mask': th.Tensor(self_mask),
                'syn_gen_mask': th.Tensor(gen_mask),
                'syn_room_types': th.Tensor(self.houses[idx][:, self.num_coords:self.num_coords+25]),
                'syn_corner_indices': th.Tensor(self.houses[idx][:, self.num_coords+25:self.num_coords+57].astype(np.uint8)),
                'syn_room_indices': th.Tensor(self.houses[idx][:, self.num_coords+57:self.num_coords+89].astype(np.uint8)),
                'syn_src_key_padding_mask': th.Tensor(src_key_padding_mask),
                'syn_connections': th.Tensor(self.houses[idx][:, self.num_coords+90:self.num_coords+92].astype(np.uint8)),
                'syn_fixed_mask': th.Tensor(self.houses[idx][:, self.num_coords+92:self.num_coords+94].astype(np.uint8)),
                'syn_graph': th.Tensor(graph),
                'graph': th.Tensor(graph),
                }
        
        return arr.transpose(0, 1), cond
    
    def __len__(self):
        return len(self.houses)

def main():
    args = create_argparser().parse_args()
    update_arg_parser(args)

    dist_util.setup_dist()
    logger.configure('ckpts/my_test')

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    
    dataset = RPlanhgDataset(length=1000)

    logger.log("sampling...")
    tmp_count = 0
    os.makedirs('outputs/pred', exist_ok=True)
    os.makedirs('outputs/gt', exist_ok=True)
    os.makedirs('outputs/gif', exist_ok=True)
    os.makedirs('outputs/graphs_gt', exist_ok=True)
    os.makedirs('outputs/graphs_pred', exist_ok=True)

    if args.dataset=='rplan':
        ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
                    6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 9: '#1F849B', 10: '#727171',
                    11: '#785A67', 12: '#D3A2C7', 13: '#FFFF00', 16: '#FFFF00', 17: '#FF00FF',
                    18: '#D3A2C7', 19: '#000000'}
        ROOM_CLASS = {"living_room": 2, "kitchen": 3, "bedroom": 4, "bathroom": 5, "balcony": 6, "entrance": 7, "dining room": 8, "study room": 9,
              "storage": 11 ,}
        
        num_room_types = 14
    else:
        print("dataset does not exist!")
        assert False
    graph_errors = []

    model_kwargs = {}
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )
    
    data_samples = []

    data_sample, model_kwarg = dataset.get_data(0)
    data_samples.append(data_sample)
    for key, value in model_kwarg.items():
        if key not in model_kwargs:
            model_kwargs[key] = [value]  # Create a new list with the value
        else:
            model_kwargs[key].append(value)  # Append to the existing list

    # print('A')
    
    for key in model_kwargs:
        model_kwargs[key] = th.stack(model_kwargs[key], 0).cuda()

    data_samples = th.stack(data_samples, 0).cuda()
        
    # Initial generation
    sample = sample_fn(
        model,
        data_samples,
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        analog_bit=args.analog_bit,
    )
    
    data_samples_update = sample[-1].clone()
    sample_gt = data_samples.unsqueeze(0)
    sample = sample.permute([0, 1, 3, 2])
    sample_gt = sample_gt.permute([0, 1, 3, 2])

    # Save initial results
    graph_error_gt, occupancy_ratios_gt = save_samples(sample_gt, 'gt', model_kwargs, tmp_count, num_room_types, ID_COLOR=ID_COLOR, is_syn=True, draw_graph=args.draw_graph, save_svg=args.save_svg, room_class=ROOM_CLASS, show_room_names=True)
    graph_error, occupancy_ratios = save_samples(sample, 'pred', model_kwargs, tmp_count, num_room_types, ID_COLOR=ID_COLOR, is_syn=True, draw_graph=args.draw_graph, save_svg=args.save_svg, save_gif=False, room_class=ROOM_CLASS, show_room_names=True)

    # Iterative generation loop
    iteration = 0
    max_iterations = 10  # Maximum number of iterations
    while iteration < max_iterations:
        print(f"\nIteration {iteration + 1}/{max_iterations}")
        
        # Ask user which rooms to fix
        print("Current layout has been saved. Please check the output images.")
        print("Enter the indices of rooms you want to fix (comma-separated, or 'done' to finish):")
        user_input = input().strip()
        
        if user_input.lower() == 'done':
            break
            
        try:
            room_indices = th.argmax(model_kwargs['syn_room_indices'], dim=-1)
            # Parse room indices to fix
            fixed_indices = [int(idx.strip()) for idx in user_input.split(',')]
            
            for room_instance in fixed_indices:
                model_kwargs['syn_src_key_padding_mask'][room_indices == room_instance] = 1
            
            # Generate new sample with fixed points
            sample = sample_fn(
                model,
                data_samples_update,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                analog_bit=args.analog_bit,
            )
            data_samples_update = sample[-1].clone()
            sample = sample.permute([0, 1, 3, 2])
            
            # Save new results
            tmp_count += 1
            graph_error, occupancy_ratios = save_samples(sample, 'pred', model_kwargs, tmp_count, num_room_types, 
                                     ID_COLOR=ID_COLOR, is_syn=True, draw_graph=args.draw_graph, 
                                     save_svg=args.save_svg, save_gif=False, room_class=ROOM_CLASS, show_room_names=True)
            
            for room_instance in fixed_indices:
                model_kwargs['syn_src_key_padding_mask'][room_indices == room_instance] = 0
            
            iteration += 1
            
        except ValueError:
            print("Invalid input. Please enter comma-separated room indices or 'done'.")
            continue

def create_argparser():
    defaults = dict(
        dataset='rplan',
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="ckpts/rplan_7/model500000.pt",
        draw_graph=True,
        save_svg=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
