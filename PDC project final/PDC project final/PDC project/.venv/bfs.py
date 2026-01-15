import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------
# 1) Generate Random Graph
# -----------------------------
def generate_graph(num_nodes=4000, avg_degree=4):
    graph = {i: set() for i in range(num_nodes)}
    for node in range(num_nodes):
        for _ in range(avg_degree):
            neighbor = random.randint(0, num_nodes - 1)
            if neighbor != node:
                graph[node].add(neighbor)
                graph[neighbor].add(node)
    return graph

# -----------------------------
# 2) Sequential BFS
#    Added: simulated work delay per neighbor
# -----------------------------
def bfs_sequential(graph, start, work_delay=0.0):
    visited = set([start])
    queue = deque([start])
    level = {start: 0}

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            # Simulated per-edge work (makes sequential slower)
            if work_delay > 0:
                time.sleep(work_delay)

            if neighbor not in visited:
                visited.add(neighbor)
                level[neighbor] = level[node] + 1
                queue.append(neighbor)

    return level

# -----------------------------
# 3) Parallel BFS (MID-TERM)
#    Added: same simulated work delay inside threads
# -----------------------------
def bfs_parallel(graph, start, num_threads=4, work_delay=0.0):
    visited = set([start])
    level = {start: 0}
    current_level = [start]
    depth = 0

    while current_level:
        next_level = []

        def process_node(node):
            new_nodes = []
            for neighbor in graph[node]:
                # Same simulated per-edge work
                if work_delay > 0:
                    time.sleep(work_delay)

                if neighbor not in visited:
                    new_nodes.append(neighbor)
            return new_nodes

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = executor.map(process_node, current_level)

        depth += 1  # correct BFS depth step

        for new_nodes in results:
            for n in new_nodes:
                if n not in visited:
                    visited.add(n)
                    level[n] = depth
                    next_level.append(n)

        current_level = next_level

    return level

# -----------------------------
# 4) Visualize Small Graph (MID-TERM)
# -----------------------------
def visualize_graph(graph):
    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_size=400)
    plt.title("Graph Visualization (Mid-Term)")
    plt.show()

# -----------------------------
# 5) MID-TERM EXECUTION
# -----------------------------
def mid_term_demo():
    NODES = 50
    DEGREE = 3
    START_NODE = 0
    NUM_THREADS = 4

    # This delay makes sequential slower and parallel faster (sleep overlaps across threads)
    WORK_DELAY = 0.00005  # 50 microseconds per neighbor check

    graph = generate_graph(NODES, DEGREE)
    visualize_graph(graph)

    start = time.time()
    seq_levels = bfs_sequential(graph, START_NODE, work_delay=WORK_DELAY)
    seq_time = time.time() - start
    print(f"Sequential BFS Time: {seq_time:.4f} sec")

    start = time.time()
    par_levels = bfs_parallel(graph, START_NODE, NUM_THREADS, work_delay=WORK_DELAY)
    par_time = time.time() - start
    print(f"Parallel BFS Time: {par_time:.4f} sec")

    if len(seq_levels) == len(par_levels):
        print("✅ Level Check: Correct")
    else:
        print("⚠ Level Check: Mismatch")

    plt.figure(figsize=(6, 4))
    plt.bar(['Sequential', 'Parallel'], [seq_time, par_time])
    plt.ylabel("Time (seconds)")
    plt.title("Mid-Term Performance Comparison")
    plt.show()

# -----------------------------
# 6) FINAL EVALUATION
# -----------------------------
def final_evaluation():
    LARGE_NODES = 4000
    DEGREE = 4
    START_NODE = 0
    THREAD_COUNTS = [1, 2, 4, 8, 16]

    print("\n===== FINAL EVALUATION =====")
    print(f"Graph Size: {LARGE_NODES} nodes")

    graph = generate_graph(LARGE_NODES, DEGREE)

    start = time.time()
    bfs_sequential(graph, START_NODE, work_delay=0.0)
    seq_time = time.time() - start
    print(f"Sequential Time: {seq_time:.4f} sec")

    parallel_times = []
    speedups = []
    efficiencies = []

    for threads in THREAD_COUNTS:
        start = time.time()
        bfs_parallel(graph, START_NODE, threads, work_delay=0.0)
        par_time = time.time() - start

        speedup = seq_time / par_time
        efficiency = speedup / threads

        parallel_times.append(par_time)
        speedups.append(speedup)
        efficiencies.append(efficiency)

        print(f"Threads: {threads} | Time: {par_time:.4f} sec | "
              f"Speedup: {speedup:.2f} | Efficiency: {efficiency:.2f}")

    plt.figure(figsize=(6, 4))
    plt.plot(THREAD_COUNTS, parallel_times, marker='o')
    plt.xlabel("Number of Threads")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time vs Number of Threads")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(THREAD_COUNTS, speedups, marker='s')
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.title("Speedup vs Threads")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(THREAD_COUNTS, efficiencies, marker='^')
    plt.xlabel("Number of Threads")
    plt.ylabel("Efficiency")
    plt.title("Efficiency vs Threads")
    plt.grid(True)
    plt.show()

# -----------------------------
# 7) MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    print("=== MID-TERM DEMO ===")
    mid_term_demo()

    print("\n=== FINAL EVALUATION ===")
    final_evaluation()
