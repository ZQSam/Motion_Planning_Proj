from random import randint,uniform,choice
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon,Point
from shapely import affinity
import heapq
import math
import argparse
from tqdm import tqdm
import time 
import concurrent.futures
import os

# Fixing tensor reshaping issue in the training script by ensuring correct data dimensions and tensor reshaping logic.

t1 = time.perf_counter()

parser = argparse.ArgumentParser()
parser.add_argument("--size",help = "size of the dataset",type=int)
parser.add_argument("--M",help= "size of the map",type=int)
parser.add_argument("--xfile",help="name the of input x dataset you're generating")
parser.add_argument("--yfile",help="name the of output y dataset you're generating")
parser.add_argument("--mode",help="enter c to create dataset and v to visualize the created dataset")
parser.add_argument("--nthread",help="number of threads you want to use",type=int)
args = parser.parse_args()

map_size = args.M
mode = args.mode
xfile = args.xfile
yfile = args.yfile
nthread = args.nthread
size = args.size
iter = range(size)

# Track if a map with no obstacles has already been generated
no_obstacle_map_generated = False

def createMapGoal(M):
  global no_obstacle_map_generated

  # Ensure only one map with no obstacles is created
  if not no_obstacle_map_generated:
    no_obstacle_map_generated = True
    O = 0  # No obstacles
  else:
    O = randint(1, 4)  # Random number of rectangular obstacles

  m = np.zeros((M, M))
  g = np.zeros((M, M))

  bounding_m = Polygon([(0, 0), (0, M), (M, M), (M, 0)])

  for _ in range(O):
    rec_length = randint(1, int(M / 2))
    rec_breadth = randint(1, int(M / 2))

    center_x = randint(0, M)
    center_y = randint(0, M)

    co_ord1 = (center_x - rec_length / 2, center_y - rec_breadth / 2)
    co_ord2 = (center_x + rec_length / 2, center_y - rec_breadth / 2)
    co_ord3 = (center_x + rec_length / 2, center_y + rec_breadth / 2)
    co_ord4 = (center_x - rec_length / 2, center_y + rec_breadth / 2)

    bounding_o = Polygon([co_ord1, co_ord2, co_ord3, co_ord4])
    angle = randint(0, 360)
    bounding_o_rot = affinity.rotate(bounding_o, angle, (center_x, center_y))

    o_map = bounding_m.intersection(bounding_o_rot)

    for row in range(M):
      for col in range(M):
        point = Point(row, col)
        if o_map.contains(point):
          m[row, col] = randint(1, 4)

  # If no obstacles were added, generate some random obstacles
  if np.all(m == 0):
    for _ in range(randint(20, 50)):
      x, y = randint(0, M-1), randint(0, M-1)
      m[x, y] = randint(1, 4)

  free_space = []

  for row in range(M):
    for col in range(M):
      if m[row, col] == 0:
        free_space.append((row, col))

  ran = free_space[randint(0, len(free_space) - 1)]
  g[ran] = 1

  return m, g, ran

def createMapGoalVisualization(m,g):
  plt.figure(figsize=(9, 3))

  plt.subplot(131)
  plt.imshow(m, cmap='viridis', vmin=0, vmax=4)
  plt.colorbar(label='Difficulty Scale')

  plt.subplot(132)
  plt.imshow(g, cmap='binary')
  plt.colorbar(label='Goal')

class Node():
    def __init__(self, parent=None, position=None,g=None):
        self.parent = parent
        self.position = position
        self.g = g

    def __lt__(self, other):
        return self.g < other.g

def createNodes(M, m):
  node_list = []
  for i in range(M):
    node_list.append([])
    for j in range(M):
      # Initialize node cost based on obstacle difficulty
      node_list[i].append(Node(parent=None,position=(i,j),g=math.inf))
      # make function 1/5-x 

  return node_list 

def isBlocked(m,point):
  return m[point] == 5

def getNeighbor(m,point,M):
  x,y = point 
  neighbor = []
  
  if x > 0:
    neighbor.append((x-1,y))
  if x < M-1:
    neighbor.append((x+1,y))
  if y > 0:
    neighbor.append((x,y-1))
  if y < M-1:
    neighbor.append((x,y+1))

  return neighbor

def Dijkstra(m,goal,M):
  Q = []
  nodes = createNodes(M, m)
  nodes[goal[0]][goal[1]].g = 0
  heapq.heappush(Q,nodes[goal[0]][goal[1]])

  while len(Q)>0:
    current = heapq.heappop(Q)
    for i in getNeighbor(m,current.position,M):
      if isBlocked(m, i):
        continue
      # Add cost of the current node's difficulty
      temp = current.g + 1 + m[i]
      if temp < nodes[i[0]][i[1]].g:
        nodes[i[0]][i[1]].g = temp
        nodes[i[0]][i[1]].parent = current

        if not nodes[i[0]][i[1]] in Q:
          heapq.heappush(Q,nodes[i[0]][i[1]])

  return nodes

def getOutput(nodes,M):
  y = np.zeros((M,M))
  for i in nodes:
    for j in i:
      y[j.position] = j.g

  return y

def createOutputVisualization(nodes,M,m):
  y = np.zeros((M,M))
  for i in nodes:
    for j in i:
      y[j.position] = j.g

  # Mark inaccessible areas (value 5) as white
  for i in range(M):
    for j in range(M):
      if m[i, j] == 5:
        y[i, j] = np.nan

  plt.subplot(133)
  plt.imshow(y, cmap='viridis')
  plt.colorbar(label='Path Cost')

# def createData(_):
#   M = map_size
#   m,g,goalcoord = createMapGoal(M)
#   x = np.stack((m,g)) #creating the input 

#   n = Dijkstra(m,goalcoord,M)
#   y = getOutput(n,M)

#   return (x,y,m,n)

def createData(_):
    m, g, goal = createMapGoal(map_size)
    x = np.stack((m, g), axis=0)  # Shape: [2, map_size, map_size]
    nodes = Dijkstra(m, goal, map_size)
    y = getOutput(nodes, map_size)  # Shape: [map_size, map_size]
    
    if np.isinf(y).any():
      print("Warning: Inf values detected in generated dataset!")
    return x, y

def generateDataset():
    all_inputs = []
    all_outputs = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=nthread) as executor:
        futures = list(tqdm(executor.map(createData, iter), total=len(iter)))
        for inp, out in futures:
            all_inputs.append(inp)
            all_outputs.append(out)

    np.savez(xfile, arr_0=np.stack(all_inputs, axis=0))  # Shape: [size, 2, map_size, map_size]
    np.savez(yfile, arr_0=np.stack(all_outputs, axis=0))  # Shape: [size, map_size, map_size]

result_dir = "Result"
os.makedirs(result_dir, exist_ok=True)

def main():
  if mode=='c':
    generateDataset()

  elif mode == 'v':
    with np.load(xfile + ".npz") as inputs, np.load(yfile + ".npz") as outputs:
        for i in range(10):  # Visualize up to 10 samples
            a = inputs[f"arr_0"][i]  # Extract a single input sample (shape: [2, map_size, map_size])
            b = outputs[f"arr_0"][i]  # Extract the corresponding output (shape: [map_size, map_size])

            m = a[0]  # Map channel
            g = a[1]  # Goal channel

            # Visualize map, goal, and output cost map
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.title("Map")
            plt.imshow(m, cmap='viridis', vmin=0, vmax=4)
            plt.colorbar(label='Difficulty Scale')

            plt.subplot(1, 3, 2)
            plt.title("Goal")
            plt.imshow(g, cmap='binary')
            plt.colorbar(label='Goal')

            plt.subplot(1, 3, 3)
            plt.title("Output (Cost Map)")
            plt.imshow(b, cmap='viridis')
            plt.colorbar(label='Path Cost')

            plt.tight_layout()
            plt.savefig(os.path.join(result_dir, f'result_{i}.png'))
            plt.show()

    t2 = time.perf_counter()
    print(f'Finished in {t2-t1} seconds')  

if __name__ == '__main__':
  main()
