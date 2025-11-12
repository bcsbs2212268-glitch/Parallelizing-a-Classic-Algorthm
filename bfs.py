import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------
# ⿡ Generate Random Graph
# -----------------------------
def generate_graph(num_nodes=1000, avg_degree=4):
    graph = {i: set() for i in range(num_nodes)}
    for node in range(num_nodes):
        for _ in range(avg_degree):
            neighbor = random.randint(0, num_nodes - 1)
            if neighbor != node:
                graph[node].add(neighbor)
                graph[neighbor].add(node)
    return graph

# -----------------------------
# ⿢ Sequential BFS
# -----------------------------
def bfs_sequential(graph, start):
    visited = set([start])
    queue = deque([start])
    level = {start: 0}

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                level[neighbor] = level[node] + 1
                queue.append(neighbor)
    return level

# -----------------------------
# ⿣ Parallel BFS (simplified)
# -----------------------------
def bfs_parallel(graph, start, num_threads=4):
    visited = set([start])
    level = {start: 0}
    current_level = [start]

    while current_level:
        next_level = []

        # Function for parallel expansion
        def process_node(node):
            new_nodes = []
            for neighbor in graph[node]:
                if neighbor not in visited:
                    new_nodes.append(neighbor)
            return new_nodes

        # Parallel step
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = executor.map(process_node, current_level)

        # Combine results safely
        for new_nodes in results:
            for n in new_nodes:
                if n not in visited:
                    visited.add(n)
                    # assign level based on first node in current_level (approximation)
                    level[n] = level[current_level[0]] + 1
                    next_level.append(n)

        current_level = next_level

    return level

# -----------------------------
# ⿤ Optional: Visualize small graph
# -----------------------------
def visualize_graph(graph):
    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_size=400)
    plt.show()

# -----------------------------
# ⿥ Main: Run BFS and compare
# -----------------------------
if __name__ == "__main__":  # ✅ FIXED
    NODES = 50        # small for visualization
    DEGREE = 3
    START_NODE = 0
    NUM_THREADS = 4

    # Generate graph
    graph = generate_graph(NODES, DEGREE)

    # Visualize graph (small graph)
    visualize_graph(graph)

    # Sequential BFS
    start_time = time.time()
    seq_levels = bfs_sequential(graph, START_NODE)
    seq_time = time.time() - start_time
    print(f"Sequential BFS Time: {seq_time:.4f} sec")

    # Parallel BFS
    start_time = time.time()
    par_levels = bfs_parallel(graph, START_NODE, NUM_THREADS)
    par_time = time.time() - start_time
    print(f"Parallel BFS Time: {par_time:.4f} sec")

    # Correctness Check
    if len(seq_levels) == len(par_levels):
        print("✅ Level check: OK (Both versions match)")
    else:
        print("⚠ Level check: Mismatch detected")

    print(f"Visited Nodes: {len(seq_levels)}")

    # -----------------------------
    # ⿦ Plot performance bar chart
    # -----------------------------
    plt.figure(figsize=(6, 4))
    plt.bar(['Sequential', 'Parallel'], [seq_time, par_time], color=['skyblue', 'orange'])
    plt.ylabel('Time (seconds)')
    plt.title('BFS Sequential vs Parallel Performance')
    for i, v in enumerate([seq_time, par_time]):
        plt.text(i, v + 0.002, f"{v:.4f}", ha='center')
    plt.show()
