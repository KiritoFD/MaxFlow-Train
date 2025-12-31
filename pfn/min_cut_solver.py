import numpy as np
from typing import List, Tuple, Set
from collections import deque

class DinicSolver:
    """
    Dinic's algorithm for maximum flow / minimum cut.
    Optimized for small graphs (< 20 nodes).
    """
    
    def __init__(self):
        self.graph = None
        self.residual = None
        self.level = None
        self.num_nodes = 0
        
    def _bfs(self, source: int, sink: int) -> bool:
        """BFS to construct level graph."""
        self.level = [-1] * self.num_nodes
        self.level[source] = 0
        queue = deque([source])
        
        while queue:
            u = queue.popleft()
            for v in range(self.num_nodes):
                if self.level[v] < 0 and self.residual[u][v] > 1e-9:
                    self.level[v] = self.level[u] + 1
                    queue.append(v)
        
        return self.level[sink] >= 0
    
    def _dfs(self, u: int, sink: int, pushed: float) -> float:
        """DFS to find augmenting paths."""
        if u == sink:
            return pushed
        
        for v in range(self.num_nodes):
            if self.level[v] == self.level[u] + 1 and self.residual[u][v] > 1e-9:
                d = self._dfs(v, sink, min(pushed, self.residual[u][v]))
                if d > 1e-9:
                    self.residual[u][v] -= d
                    self.residual[v][u] += d
                    return d
        
        return 0
    
    def solve(self, capacity: np.ndarray, source: int, sink: int) -> Tuple[float, np.ndarray]:
        """
        Compute maximum flow using Dinic's algorithm.
        
        Args:
            capacity: Capacity matrix (n x n)
            source: Source node index
            sink: Sink node index
            
        Returns:
            max_flow: Maximum flow value
            residual: Residual graph after max flow
        """
        self.num_nodes = capacity.shape[0]
        self.residual = capacity.copy().astype(float)
        
        max_flow = 0.0
        
        while self._bfs(source, sink):
            while True:
                pushed = self._dfs(source, sink, float('inf'))
                if pushed <= 1e-9:
                    break
                max_flow += pushed
        
        return max_flow, self.residual
    
    def find_min_cut(self, capacity: np.ndarray, source: int, sink: int) -> Tuple[float, List[Tuple[int, int]], Set[int], Set[int]]:
        """
        Find minimum cut edges.
        
        Returns:
            max_flow: Maximum flow value (equals min cut capacity)
            cut_edges: List of (u, v) edges in the minimum cut
            S_set: Nodes reachable from source in residual graph
            T_set: Nodes not reachable from source
        """
        max_flow, residual = self.solve(capacity, source, sink)
        
        # Find S-set: nodes reachable from source in residual graph
        S_set = set()
        queue = deque([source])
        S_set.add(source)
        
        while queue:
            u = queue.popleft()
            for v in range(self.num_nodes):
                if v not in S_set and residual[u][v] > 1e-9:
                    S_set.add(v)
                    queue.append(v)
        
        T_set = set(range(self.num_nodes)) - S_set
        
        # Find cut edges: edges from S to T with positive original capacity
        cut_edges = []
        for u in S_set:
            for v in T_set:
                if capacity[u][v] > 1e-9:
                    cut_edges.append((u, v))
        
        return max_flow, cut_edges, S_set, T_set
    
    def get_bottleneck_layers(self, cut_edges: List[Tuple[int, int]], 
                              graph_builder) -> List[str]:
        """
        Identify bottleneck layer blocks from cut edges.
        
        Returns:
            List of block names that form the bottleneck
        """
        bottlenecks = []
        for u, v in cut_edges:
            u_name = graph_builder.get_node_name(u)
            v_name = graph_builder.get_node_name(v)
            bottlenecks.append(f"{u_name} -> {v_name}")
        return bottlenecks
