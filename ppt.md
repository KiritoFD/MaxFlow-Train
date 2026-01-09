---
theme: seriph
background: https://source.unsplash.com/collection/94734566/1920x1080
class: 'text-center'
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
transition: slide-left
title: 基于最大流最小割的神经网络优化
mdc: true
---

# 基于最大流最小割的神经网络梯度流瓶颈诊断与拓扑优化

<div class="pt-12">
  <span @click="$slidev.nav.next" class="px-2 py-1 rounded cursor-pointer" hover="bg-white bg-opacity-10">
    肖阳 (24300240055) <br>
    《集合与图论》课程论文汇报
  </span>
</div>

<div class="abs-br m-6 flex gap-2">
  <a href="https://github.com/slidevjs/slidev" target="_blank" alt="Slidev"
    class="text-xl slidev-icon-btn opacity-50 !border-none !hover:text-white">
    <carbon-logo-github />
  </a>
</div>

---
layout: default
---

# 1. 现有困境：优化的“盲人摸象”

<div class="grid grid-cols-2 gap-4">

<div>

### 传统视角 (SGD/Adam)
- **点感知 (Point-wise)**：仅关注单个参数的一阶/二阶矩。
- **假设**：参数之间是独立的统计分布。
- **缺陷**：无法感知网络的**拓扑连通性**。

</div>

<div>

### 图论视角 (Our Approach)
- **边感知 (Edge-wise)**：关注梯度在层级间的流动。
- **假设**：网络是一个有向无环流网络。
- **优势**：利用 **最小割 (Min-Cut)** 定位全局瓶颈。

</div>

</div>

<br>
<br>

> **核心冲突**：当网络出现结构性瓶颈（如深层梯度消失）时，盲目调整局部学习率就像**“在堵车的非瓶颈路段加速”**，对提升全局通量无效。

---
layout: two-cols
---

# 2. 理论建模：从 NN 到流网络

我们将神经网络映射为有向图 $G=(V, E, c)$：

<v-clicks>

- **节点 $V$ (Block)**：
  为了满足流的定义，实施**显式分块 (Explicit Partitioning)**。将一组神经元聚合为图论中的一个节点 $v_i$。

- **边 $E$ (Flow)**：
  反向传播中的梯度流 $\nabla \mathcal{L}$。
  <br> $Loss \to Source, Input \to Sink$。

- **容量 $c(e)$ (Capacity)**：
  定义为分块梯度的 **$L_2$ 范数**。
  $$c(u, v) = \mathbb{E}[\|\nabla_{block} \mathcal{L}\|_2]$$
  这代表了该路径承载梯度更新能量的物理上限。

</v-clicks>

::right::

<div class="ml-4 mt-10 p-4 bg-gray-100 rounded-lg dark:bg-gray-800">

### 引理 1：流量守恒性
在分块尺度下，梯度流严格遵循 Kirchhoff 定律：

$$\sum_{u \in in(v)} f(u, v) = \sum_{w \in out(v)} f(v, w)$$

这意味着我们可以直接套用 **Max-Flow Min-Cut** 定理。

</div>

---

# 3. 核心定理：非饱和路径的无效性

为什么我们要找最小割？基于 **残余网络 (Residual Network)** 的分析：

<div class="bg-blue-50 dark:bg-blue-900 p-4 rounded-lg my-4">

**引理 3**：在最小割 $E_{cut}$ 状态不变的条件下，增加任意非饱和边 $e' \notin E_{cut}$ 的容量，对提升系统全局通量 $\Phi$ **无益**。

</div>

<div class="grid grid-cols-2 gap-8">

<div>

**证明逻辑：**
1. 最大流值 $\Phi_{max} = C(S, T)$。
2. 非饱和边 $e'$ 不在割集求和项中。
3. 在 $G_f$ 中，增加 $c(e')$ 无法构建从 $S$ 到 $T$ 的增广路径。

</div>

<div>

**推论：**
- 只有疏通 **最小割 (Min-Cut)** 才能提升收敛速度。
- 传统的全局学习率调整是在做大量的“无效功”。

</div>

</div>

---
layout: two-cols
---

# 4. PFN 算法框架

**参数流网络 (Parameter Flow Network)**

1.  **构建流图**：实时计算各 Block 的梯度范数 $c(e)$。
2.  **求解最大流**：使用 **Push-Relabel** 算法 ($O(V^2E)$)。
    - *注：由于 $|V| \approx 10^2$，该算法开销可忽略。*
3.  **对偶诊断**：利用 $S-T$ 割提取最小割集 $E_{cut}$。
4.  **拓扑补偿**：
    $$\eta_{new} = \eta \cdot (1 + \gamma \cdot \mathbb{I}(e \in E_{cut}))$$

::right::

```python {all|2-3|5-6|8-9}
# PFN 伪代码逻辑
def train_step(model, data):
    # 1. 前向 + 反向传播
    loss.backward()
    
    # 2. 构建图并求解最小割
    graph = build_flow_graph(model)
    min_cut_edges = push_relabel(graph)
    
    # 3. 仅对割边进行梯度补偿
    for block in model.blocks:
        if block in min_cut_edges:
            block.grad *= boost_factor
            
    # 4. 优化器更新
    optimizer.step()
5. 实验结果：No-BN 场景的鲁棒性
<div class="grid grid-cols-2 gap-4">

<div class="h-60 bg-gray-200 flex items-center justify-center rounded text-gray-500"> 此处插入实验图 1 (Baseline vs PFN in No-BN) </div>

<div class="h-60 bg-gray-200 flex items-center justify-center rounded text-gray-500"> 此处插入实验图 2 (Min-Cut 随 Epoch 的分布演化) </div>

</div>

现象：在移除 BatchNorm 后，Baseline 模型（蓝线）迅速崩溃，梯度消失。

PFN 表现：PFN（红线）通过动态识别并补偿 fc_block 等瓶颈，维持了接近标准架构的性能。

瓶颈转移：日志显示最小割在 Conv 层与 FC 层间轮动，验证了算法的动态调节能力。

layout: center class: text-center
结论
<div class="text-left max-w-2xl mx-auto text-xl">

视角转换：神经网络不仅是代数计算图，更是拓扑流网络。

理论支撑：最大流最小割定理为深度学习的瓶颈诊断提供了数学完备的解释。

工程价值：PFN 框架证明了，通过图论方法可以精准定位并突破训练中的“结构化窄喉”。

</div>

<span class="text-sm opacity-50"> Keywords: Max-Flow Min-Cut, Gradient Flow, Push-Relabel Algorithm, Topology Optimization </span>