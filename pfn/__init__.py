from .PFN import PFNGraphBuilder, IncrementalPushRelabel, BottleneckOptimizer

# 向后兼容别名
DinicSolver = IncrementalPushRelabel

__all__ = ['PFNGraphBuilder', 'IncrementalPushRelabel', 'DinicSolver', 'BottleneckOptimizer']
