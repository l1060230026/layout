"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import multiprocessing as mp
from functools import partial

import matplotlib
matplotlib.use('Agg')
import numpy as np
import math
import torch as th

import io
import PIL.Image as Image
import drawSvg as drawsvg
import cairosvg
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_fid.fid_score import calculate_fid_given_paths
from ConDiffPlan.rplanhg_datasets import load_rplanhg_data
from ConDiffPlan.msd_datasets import load_msd_data
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
from shapely.geometry import LineString

# import random
# th.manual_seed(0)
# random.seed(0)
# np.random.seed(0)

def get_edges(polygon):
    """Get all edges of the polygon"""
    coords = list(polygon.exterior.coords)
    edges = []
    for i in range(len(coords) - 1):
        edges.append(LineString([coords[i], coords[i + 1]]))
    return edges

def line_segments_distance(line1, line2):
    """Calculate the minimum distance between two line segments"""
    # If line segments intersect, distance is 0
    if line1.intersects(line2):
        return 0
    
    # Get start and end coordinates of line segments
    p1, p2 = np.array(line1.coords)
    p3, p4 = np.array(line2.coords)
    
    # Calculate direction vectors of line segments
    v1 = p2 - p1
    v2 = p4 - p3
    
    # Check if line segments are parallel (by checking if cross product is close to 0)
    cross_product = np.cross(v1, v2)
    if abs(cross_product) < 1e-10:  # Parallel
        # Calculate distance from points of first line segment to second line segment
        distances = [
            point_to_line_segment_distance(p1, p3, p4),
            point_to_line_segment_distance(p2, p3, p4),
            point_to_line_segment_distance(p3, p1, p2),
            point_to_line_segment_distance(p4, p1, p2)
        ]
        return min(distances)
    
    return line1.distance(line2)

def point_to_line_segment_distance(p, line_start, line_end):
    """Calculate the distance from a point to a line segment"""
    line_vec = line_end - line_start
    point_vec = p - line_start
    line_length = np.linalg.norm(line_vec)
    line_unit_vec = line_vec / line_length
    projection_length = np.dot(point_vec, line_unit_vec)
    
    if projection_length < 0:
        return np.linalg.norm(point_vec)
    elif projection_length > line_length:
        return np.linalg.norm(p - line_end)
    else:
        projection = line_start + line_unit_vec * projection_length
        return np.linalg.norm(p - projection)

def polygon_parallel_distance(poly1, poly2):
    """Calculate the distance between parallel edges of two polygons"""
    edges1 = get_edges(poly1)
    edges2 = get_edges(poly2)
    
    min_distance = float('inf')
    for edge1 in edges1:
        for edge2 in edges2:
            dist = line_segments_distance(edge1, edge2)
            min_distance = min(min_distance, dist)
    
    return min_distance

bin_to_int = lambda x: int("".join([str(int(i.cpu().data)) for i in x]), 2)
def bin_to_int_sample(sample, resolution=256):
    sample_new = th.zeros([sample.shape[0], sample.shape[1], sample.shape[2], 2])
    sample[sample<0] = 0
    sample[sample>0] = 1
    for i in range(sample.shape[0]):
        for j in range(sample.shape[1]):
            for k in range(sample.shape[2]):
                sample_new[i, j, k, 0] = bin_to_int(sample[i, j, k, :8])
                sample_new[i, j, k, 1] = bin_to_int(sample[i, j, k, 8:])
    sample = sample_new
    sample = sample/(resolution/2) - 1
    return sample

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
        if m > 0 and (_type_k not in [16, 18, 19]) and (_type_l not in [16, 18, 19]):
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
    # G_gt = G_gt[1-th.where((G_gt == th.tensor([0,0,0], device='cuda')).all(dim=1))[0]]
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
    doors_inds = np.where((nodes == 16) | (nodes == 18))[0]
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
                # Check intersection and boundary distance
                iou = p1.intersection(p2).area / p1.union(p2).area
                # boundary_distance = polygon_parallel_distance(p1, p2)
                if (iou > 0 and iou < 0.2):  # Add boundary distance judgment
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

def save_samples(
        sample, ext, model_kwargs, 
        tmp_count, num_room_types, 
        save_gif=False, save_edges=False,
        door_indices = [16, 18, 19], ID_COLOR=None,
        is_syn=False, draw_graph=False, save_svg=False):
    prefix = 'syn_' if is_syn else ''
    graph_errors = []
    if not save_gif:
        sample = sample[-1:]
    for i in range(sample.shape[1]):
        resolution = 512
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
            for j, point in (enumerate(sample[k][i])):
                if model_kwargs[f'{prefix}gen_mask'][i][0][j]==1:
                    continue
                point = point.cpu().data.numpy()
                if j==0:
                    poly = []
                if j>0 and (model_kwargs[f'{prefix}room_indices'][i, j]!=model_kwargs[f'{prefix}room_indices'][i, j-1]).any():
                    polys.append(poly)
                    types.append(c)
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
            polys.append(poly)
            types.append(c)
            for poly, c in zip(polys, types):
                if c in door_indices or c==0:
                    continue
                room_type = c
                c = webcolors.hex_to_rgb(ID_COLOR[c])
                draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='black', stroke_width=1))
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
                # draw_color.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill='none', fill_opacity=0.0, stroke='black', stroke_width=1))
                draw.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill='black', fill_opacity=0.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                draw2.append(drawsvg.Lines(*np.array(poly).flatten().tolist(), close=True, fill=ID_COLOR[room_type], fill_opacity=1.0, stroke=webcolors.rgb_to_hex([int(x/2) for x in c]), stroke_width=0.5*(resolution/256)))
                for corner in poly:
                    draw.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
                    draw3.append(drawsvg.Circle(corner[0], corner[1], 2*(resolution/256), fill=ID_COLOR[room_type], fill_opacity=1.0, stroke='gray', stroke_width=0.25))
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
        
        for image in images:
            image.close()
        for image in images2:
            image.close()
        for image in images3:
            image.close()
    
    return graph_errors

def save_samples_process(sample, ext, model_kwargs, tmp_count, num_room_types, 
                        save_gif=False, save_edges=False,
                        door_indices=[16, 18, 19], ID_COLOR=None,
                        is_syn=False, draw_graph=False, save_svg=False):
    """Process function to save samples in a separate process"""
    try:
        return save_samples(sample, ext, model_kwargs, tmp_count, num_room_types,
                          save_gif, save_edges, door_indices, ID_COLOR,
                          is_syn, draw_graph, save_svg)
    except Exception as e:
        print(f"Error in save_samples_process: {e}")
        return []

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

    errors = []
    for _ in range(1):
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
            num_room_types = 14
            args.target_set = 5
            data = load_rplanhg_data(
                batch_size=args.batch_size,
                analog_bit=args.analog_bit,
                set_name='eval',
                target_set=args.target_set,
                rtype_dim=args.rtype_dim,
                corner_index_dim=args.corner_index_dim,
                room_index_dim=args.room_index_dim,
                max_num_points=getattr(args, 'max_num_points', 200),
            )
        elif args.dataset=='msd':
            ID_COLOR = {1: '#EE4D4D', 2: '#C67C7B', 3: '#FFD274', 4: '#BEBEBE', 5: '#BFE3E8',
                        6: '#7BA779', 7: '#E87A90', 8: '#FF8C69', 9: '#1F849B'}
            num_room_types = 9
            data = load_msd_data(
                batch_size=args.batch_size,
                analog_bit=args.analog_bit,
                set_name='eval',
                rtype_dim=args.rtype_dim,
                corner_index_dim=args.corner_index_dim,
                room_index_dim=args.room_index_dim,
                max_num_points=getattr(args, 'max_num_points', 200),
            )
        else:
            print("dataset does not exist!")
            assert False
        graph_errors = []
        for tmp_count in tqdm(range(0, args.num_samples, args.batch_size)):
            model_kwargs = {}
            sample_fn = (
                diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
            )
            data_sample, data_sample_syn, model_kwargs = next(data)
            for key in model_kwargs:
                model_kwargs[key] = model_kwargs[key].cuda()
                
            # room_indices = th.argmax(model_kwargs['syn_room_indices'], dim=-1).cpu()
            # for k in range(1, args.target_set):

            #     sample = sample_fn(
            #         model,
            #         data_sample_syn,
            #         clip_denoised=args.clip_denoised,
            #         model_kwargs=model_kwargs,
            #         analog_bit=args.analog_bit,
            #     )
                
            #     sample = sample[-1].cpu()
                
            #     data_sample_syn.permute([0, 2, 1])[room_indices==k] = sample.permute([0, 2, 1])[room_indices==k]
                
            #     model_kwargs['syn_src_key_padding_mask'][room_indices == k] = 1
            #     model_kwargs['syn_fixed_mask'][room_indices == k] = 1
                
            
            sample = sample_fn(
                    model,
                    data_sample_syn,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    analog_bit=args.analog_bit,
                )
                
            sample_gt = data_sample.cuda().unsqueeze(0)
            sample = sample.permute([0, 1, 3, 2])
            sample_gt = sample_gt.permute([0, 1, 3, 2])
            if args.analog_bit:
                sample_gt = bin_to_int_sample(sample_gt)
                sample = bin_to_int_sample(sample)

            # Move tensors to CPU before passing to the process
            sample_gt_cpu = sample_gt.cpu()
            sample_cpu = sample.cpu()
            model_kwargs_cpu = {k: v.cpu() for k, v in model_kwargs.items()}
            
            if args.debug:
                # Single-threaded debug mode: directly call function for easier error stack viewing
                print("Running in debug mode (single-threaded)")
                gt_errors = save_samples_process(sample_gt_cpu, 'gt', model_kwargs_cpu, tmp_count, 
                                               num_room_types, False, False, [16, 18, 19], 
                                               ID_COLOR, False, args.draw_graph, args.save_svg)
                
                pred_errors = save_samples_process(sample_cpu, 'pred', model_kwargs_cpu, tmp_count,
                                                  num_room_types, False, False, [16, 18, 19],
                                                  ID_COLOR, True, args.draw_graph, args.save_svg)
                
                graph_errors.extend(pred_errors)
            else:
                # Multi-process mode: parallel processing to improve speed
                with mp.Pool(processes=2) as pool:
                    # Run the processes in parallel
                    gt_async = pool.apply_async(save_samples_process, 
                                         args=(sample_gt_cpu, 'gt', model_kwargs_cpu, tmp_count, 
                                              num_room_types, False, False, [16, 18, 19], 
                                              ID_COLOR, False, args.draw_graph, args.save_svg))
                    
                    pred_async = pool.apply_async(save_samples_process,
                                           args=(sample_cpu, 'pred', model_kwargs_cpu, tmp_count,
                                                num_room_types, False, False, [16, 18, 19],
                                                ID_COLOR, True, args.draw_graph, args.save_svg))
                    
                    # Get results
                    gt_errors = gt_async.get()
                    pred_errors = pred_async.get()
                    
                    graph_errors.extend(pred_errors)
        logger.log("sampling complete")
        fid_score = calculate_fid_given_paths(['outputs/gt', 'outputs/pred'], 64, 'cuda', 2048)
        print(f'FID: {fid_score}')
        print(f'Compatibility: {np.mean(graph_errors)}')
        errors.append([fid_score, np.mean(graph_errors)])
    errors = np.array(errors)
    print(f'Diversity mean: {errors[:, 0].mean()} \t Diversity std: {errors[:, 0].std()}')
    print(f'Compatibility mean: {errors[:, 1].mean()} \t Compatibility std: {errors[:, 1].std()}')

def create_argparser():
    defaults = dict(
        dataset='rplan',
        clip_denoised=True,
        num_samples=10000,
        batch_size=8,
        use_ddim=False,
        model_path="ckpts/rplan_7/model500000.pt",
        draw_graph=True,
        save_svg=False,
        debug=True,  # Single-threaded debug mode
        # Dataset hyperparameters
        rtype_dim=25,
        corner_index_dim=32,
        room_index_dim=32,
        max_num_points=100,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
