# Import for I/O

import random
a = random.randint(0,10000000000)

import matplotlib.pyplot as plt
import math
from functools import partial
import seaborn as sns
import numpy as np
import time
import csv

from gerrychain import Graph, updaters
from gerrychain.updaters import Tally, cut_edges
from gerrychain.partition import Partition
import networkx as nx
import geopandas as gpd
from math import sin, cos, sqrt, atan2, radians
from shapely.geometry import Polygon

########################### Functions ###############################


def gdf_print_map(partition, filename, gdf, unit_name):
    #generate map figures from geodatagrame
    cdict = {partition.graph.nodes[i][unit_name]:partition.assignment[i] for i in partition.graph.nodes()}
    gdf['color'] = gdf.apply(lambda x: cdict[x[unit_name]], axis=1)
    plt.figure()
    gdf.plot(column='color')
    plt.savefig(filename+'.png')
    plt.close()


def centroid_dist_euclidean(u,v):
    return math.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)

def centroid_dist_lat_lon(u,v):
    R = 6373.0

    lat1 = radians(u[1])
    lon1 = radians(u[0])
    lat2 = radians(v[1])
    lon2 = radians(v[0])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


def merge_parts_chen(partition, k, dist_func, draw_map = False, unit_key = 'GEOID10'):
    #based on Chen's iterative merging technique described in "The Loser’s Bonus: Political Geography and Minority Party Representation"
    
    assert (len(partition.parts) >= k)
    #until there are k parts, choose part randomly and merge with closest part
    while len(partition.parts) > k:
        part_rand = random.choice(range(len(partition.parts)))
        closest_part = (part_rand + 1) % len(partition.parts)
        dist_min = math.inf
        for e in partition["cut_edges"]:
            if partition.assignment[e[0]] == part_rand:
                i = partition.assignment[e[1]]
                dist_i = dist_func(partition.centroids[part_rand], partition.centroids[i])
                if dist_i < dist_min:
                    closest_part = i
                    dist_min = dist_i
            if partition.assignment[e[1]] == part_rand:
                i = partition.assignment[e[0]]
                dist_i = dist_func(partition.centroids[part_rand], partition.centroids[i])
                if dist_i < dist_min:
                    closest_part = i
                    dist_min = dist_i
        if draw_map:
            gdf_print_map(partition, './figs/iter_merge_'+str(len(partition.parts))+'.png', gdf, unit_key)
        merge_dict = {v:part_rand if partition.assignment[v] == closest_part else partition.assignment[v] for v in partition.graph.nodes()}
        partition = Partition(partition.graph, merge_dict, partition.updaters)
        keyshift = dict(zip(list(partition.parts.keys()), range(len(partition.parts.keys()))))
        keydict = {v:keyshift[partition.assignment[v]] for v in partition.graph.nodes()}
        partition = Partition(partition.graph, keydict, partition.updaters)
    return partition


def shift_chen(partition, ep, rep_max, ideal_pop, dist_func, draw_map= False):
    #based on Chen's population rebalancing technique described in "The Loser’s Bonus: Political Geography and Minority Party Representation"

    counter = 0
    past_10 = []
    while max([abs(partition["population"][i]-ideal_pop) for i in dict(partition["population"]).keys()]) > ep*ideal_pop and counter < rep_max:
        if len(past_10) == 10 and len(set(past_10)) <=2:        #check for infinite (two-cycle) loop
            return partition
        #identify adjacent pair of districts with largest population difference
        max_diff_pair = (partition.assignment[list(partition["cut_edges"])[0][0]], partition.assignment[list(partition["cut_edges"])[0][1]])
        max_diff_score = 0
        max_diff_edges = []
        for e in partition["cut_edges"]:
            score =  abs(partition["population"][partition.assignment[e[0]]] - partition["population"][partition.assignment[e[1]]])
            if score > max_diff_score:
                max_diff_edges = [e]
                max_diff_score = score
                max_diff_pair = (partition.assignment[e[0]], partition.assignment[e[1]])
            elif partition.assignment[e[0]] in max_diff_pair and partition.assignment[e[1]] in max_diff_pair:
                max_diff_edges.append(e)
        if partition["population"][max_diff_pair[0]] >= partition["population"][max_diff_pair[1]]:
            unit_m = max_diff_pair[0]
            unit_l = max_diff_pair[1]
        else:
            unit_m = max_diff_pair[1]
            unit_l = max_diff_pair[0]

        #identify set of units that can be moved from overpopulated district to underpopulated district
        moveable_units = {}
        for e in max_diff_edges:
            if partition.assignment[e[0]] == unit_m:
                assert(partition.assignment[e[1]] == unit_l)
                edge_unit_max = e[0]
                edge_unit_min = e[1]
            else:
                assert(partition.assignment[e[0]] == unit_l)
                assert(partition.assignment[e[1]] == unit_m)
                edge_unit_max = e[1]
                edge_unit_min = e[0]
            #check contiguity of proposed move
            subg = partition.graph.subgraph(partition.parts[partition.assignment[edge_unit_max]]).copy()
            subg.remove_node(edge_unit_max)
            if nx.is_connected(subg):
                unit_centroid = (float(partition.graph.nodes[edge_unit_max][x_name]), float(partition.graph.nodes[edge_unit_max][y_name]))
                moveable_units[(edge_unit_max, edge_unit_min)] = dist_func(partition.centroids[unit_m], unit_centroid) - dist_func(partition.centroids[unit_l], unit_centroid)
        
        if len(moveable_units) == 0:
            return partition
        #move unit with largest difference between distance to larger district's centroid and distance to smaller district's centroid
        max_dp = max(moveable_units.values())
        move_unit = [i for i in moveable_units.keys() if moveable_units[i] == max_dp][0]
        if len(past_10) >= 10:
            past_10 = past_10[-9:]
        past_10.append((move_unit[0], unit_m))
        shift_dict = {v:unit_l if v == move_unit[0] else partition.assignment[v] for v in partition.graph.nodes()}
        partition = Partition(partition.graph, shift_dict, partition.updaters)
        if draw_map:
            gdf_print_map(partition, './figs/iter_merge_pop_shift'+str(counter)+'.png', gdf, unit_key)
        counter += 1
    return partition 


def record_partition(partition, t, run_type):
    with open('./output/Tex'+ graph_name+".txt", 'a+') as partition_file:
        writer = csv.writer(partition_file)
        out_row = [time.time(), graph_name, run_type, t, len(graph.nodes())]+[partition.assignment[x] for x in graph.nodes()]
        writer.writerow(out_row)   


def centroids(partition):
    CXs = {k: partition["Sum_areaCX"][k]/partition["Sum_area"][k] for k in list(partition.parts.keys())}
    CYs = {k: partition["Sum_areaCY"][k]/partition["Sum_area"][k] for k in list(partition.parts.keys())}
    centroids = {k: (CXs[k], CYs[k]) for k in list(partition.parts.keys())}
    return centroids


# ######################### Iowa Setup #########################

# graph_name = 'iowa'
# graph_path = './state_data/'+graph_name+'.json'
# graph = Graph.from_json(graph_path)
# k=4
# shapefile_name = 'IA_counties'
# gdf = gpd.read_file('./state_data/'+shapefile_name)
# gdf = gdf.to_crs({'init': 'epsg:26775'})


# node_list = list(graph.nodes())
# edge_list = list(graph.edges())
# num_counties = len(node_list)
# ideal_pop = sum([graph.nodes[v]["TOTPOP"] for v in graph.nodes()])/k
# ep = 0.05

# area_name = 'area'
# x_name = 'INTPTLON10'
# y_name = 'INTPTLAT10'
# areaC_X = "areaC_X"
# areaC_Y = "areaC_Y"
# area = "area"
# unit_key = 'GEOID10'

# for node in graph.nodes():
#     graph.nodes[node]["areaC_X"] = float(graph.nodes[node][area_name])*float(graph.nodes[node][x_name])
#     graph.nodes[node]["areaC_Y"] = float(graph.nodes[node][area_name])*float(graph.nodes[node][y_name])
#     graph.nodes[node]["area"] = float(graph.nodes[node][area_name])


# # Necessary updaters go here
# updaters = {
#     "population": Tally("TOTPOP", alias="population"),
#     "cut_edges": cut_edges,
#     "Sum_areaCX": Tally(areaC_X, alias = "Sum_areaCX"),
#     "Sum_areaCY": Tally(areaC_Y, alias = "Sum_areaCY"),
#     "Sum_area": Tally(area, alias = "Sum_area"),
#     "centroids": centroids
# }

# cddict = {v:v for v in graph.nodes()}
# part_all_units = Partition(graph, assignment=cddict, updaters=updaters)


# ################# Iowa experiment  #################

# data_dir = './output_data/'
# run_type = 'iowa'
# num_runs = 10000
# print_step = num_runs/100
# rep_max = 10000  #max number of unit shifts attempted in population shift before failure

# with open(data_dir + "iowa_iterative_merge_cuts.txt", 'a+') as iowa_merge_cuts:
#     writer = csv.writer(iowa_merge_cuts)
                 
#     successes = 0
#     cuts = []
#     for i in range(num_runs):
#         if i%print_step == 0:
#             print(i)
#         part_merge = merge_parts_chen(part_all_units , k, centroid_dist_lat_lon)
#         part_shift = shift_chen(part_merge, ep, rep_max, ideal_pop, centroid_dist_lat_lon)
#         if max([abs(part_shift["population"][i]-ideal_pop) for i in dict(part_shift["population"]).keys()]) <= ep*ideal_pop:
#             successes += 1
#             cuts.append(list(part_shift['cut_edges']))
#             outrow = [time.time(), run_type, i, successes, ep, num_runs, len(list(part_shift['cut_edges']))] + list(part_shift['cut_edges'])
#             writer.writerow(outrow)


######################## 10 x 10 grid setup ##################################

graph_name = 'grid'
k=4
grid = nx.grid_graph([10,10])
grid_pop_map = {v:1 for v in grid.nodes()}
grid_pos_map= {v:v for v in grid.nodes()}
grid_ideal_pop = 25
node_list = list(grid.nodes())
edge_list = list(grid.edges())


for v in grid.nodes():
    grid.nodes[v]["TOTPOP"] = 1
    grid.nodes[v]["area"] = 1
    grid.nodes[v]["x"] = v[0]
    grid.nodes[v]["y"] = v[1]
    grid.nodes[v]["g_nodes"] = v


rows = 10
cols = 10
width = 1
height = 1
xmin,ymin,xmax,ymax =  0,0, cols*width, rows*height

XleftOrigin = xmin
XrightOrigin = xmin + width
YtopOrigin = ymax
YbottomOrigin = ymax- height
polygons = []
for i in range(cols):
    Ytop = YtopOrigin
    Ybottom =YbottomOrigin
    for j in range(rows):
        polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)])) 
        Ytop = Ytop - height
        Ybottom = Ybottom - height
    XleftOrigin = XleftOrigin + width
    XrightOrigin = XrightOrigin + width

x_vals = [j for i in range(cols) for j in range(rows)]
y_vals = [i for i in range(cols) for j in range(rows)]

gdf = gpd.GeoDataFrame({'geometry':polygons, 'x':x_vals, 'y': y_vals, 'g_nodes':list(grid.nodes())})


ideal_pop = sum([grid.nodes[v]["TOTPOP"] for v in grid.nodes()])/k
ep = 0.05

cddict = {list(grid.nodes())[i]:i for i in range(len(grid.nodes()))}

area_name = 'area'
x_name = 'x'
y_name = 'y'
areaC_X = "areaC_X"
areaC_Y = "areaC_Y"
area = "area"
unit_key = 'g_nodes'

for node in grid.nodes():
    grid.nodes[node]["areaC_X"] = float(grid.nodes[node][area_name])*float(grid.nodes[node][x_name])
    grid.nodes[node]["areaC_Y"] = float(grid.nodes[node][area_name])*float(grid.nodes[node][y_name])
    grid.nodes[node]["area"] = float(grid.nodes[node][area_name])


# Necessary updaters go here
updaters = {
    "population": Tally("TOTPOP", alias="population"),
    "cut_edges": cut_edges,
    "Sum_areaCX": Tally(areaC_X, alias = "Sum_areaCX"),
    "Sum_areaCY": Tally(areaC_Y, alias = "Sum_areaCY"),
    "Sum_area": Tally(area, alias = "Sum_area"),
    "centroids": centroids
}

part_all_units = Partition(grid, assignment=cddict, updaters=updaters)


################# grid experiment #################

data_dir = './output_data/'
run_type = 'grid'
num_runs = 10000
print_step = num_runs/100
rep_max = 10000  #max number of unit shifts attempted in population shift before failure

with open(data_dir + "grid_iterative_merge_cuts.txt", 'a+') as grid_merge_cuts:
    writer = csv.writer(grid_merge_cuts)
                 
    successes = 0
    cuts = []
    for i in range(num_runs):
        if i%print_step == 0:
            print(i)
        part_merge = merge_parts_chen(part_all_units , k, centroid_dist_euclidean)
        part_shift = shift_chen(part_merge, ep, rep_max, ideal_pop, centroid_dist_euclidean)
        if max([abs(part_shift["population"][i]-ideal_pop) for i in dict(part_shift["population"]).keys()]) <= ep*ideal_pop:
            successes += 1
            cuts.append(list(part_shift['cut_edges']))
            outrow = [time.time(), run_type, i, successes, ep, num_runs, len(list(part_shift['cut_edges']))] + list(part_shift['cut_edges'])
            writer.writerow(outrow)
