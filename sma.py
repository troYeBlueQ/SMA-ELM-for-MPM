import numpy as np

def initialize_positions(population_size, dim, lb, ub):
    """
       Initialize the positions of search agents.

       Parameters:
       - SearchAgents_no: Number of search agents
       - dim: Dimension of the search space
       - ub: Upper boundary (can be scalar or list/array for each dimension)
       - lb: Lower boundary (can be scalar or list/array for each dimension)

       Returns:
       - Positions: Initialized positions of search agents as a numpy array
       """
    # Determine the number of boundaries
    Boundary_no = len(ub) if isinstance(ub, (list, np.ndarray)) else 1

    # Initialize positions based on the type of boundaries
    if Boundary_no == 1:
        # If boundaries are uniform for all dimensions
        Positions = np.random.rand(population_size, dim) * (ub - lb) + lb
    else:
        # If boundaries differ for each dimension
        Positions = np.zeros((population_size, dim))
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            Positions[:, i] = np.random.rand(population_size) * (ub_i - lb_i) + lb_i

    return Positions

def SMA(fitness_function, bounds, population_size, max_iterations):
    """
    黏菌优化算法核心实现。

    Args:
        fitness_function: 可调用的适应度函数，输入参数数组返回适应度值。
        bounds: 参数搜索范围，列表 [(lower, upper), ...]。
        population_size: 种群规模。
        max_iterations: 最大迭代次数。

    Returns:
        best_position: 全局最优解的位置。
        best_fitness: 全局最优适应度值。
        convergence_curve: 收敛曲线（每次迭代的最优适应度值）。
        vb_vc_history: 随机个体的 vb 和 vc 记录。
    """
    dim = len(bounds)  # 参数维度
    lb, ub = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
    positions = initialize_positions(population_size, dim, lb, ub)  # 初始化种群位置
    best_position = np.zeros(dim)
    best_fitness = float('inf')
    convergence_curve = []
    vb_vc_history = []  # 用于记录随机个体的 vb 和 vc 值

    # 随机选定一个个体索引
    random_individual_index = np.random.randint(0, population_size)

    for iteration in range(max_iterations):
        # 计算所有个体的适应度值
        fitness = np.apply_along_axis(fitness_function, 1, positions)
        sorted_indices = np.argsort(fitness)  # 按适应度排序
        best_index = sorted_indices[0]  # 当前最优个体索引

        if fitness[best_index] < best_fitness:
            best_fitness = fitness[best_index]
            best_position = positions[best_index]

        # 动态调整参数 a 和 b
        a = np.arctanh(np.clip(-iteration / max_iterations + 1, -0.999, 0.999))
        b = 1 - iteration / max_iterations

        for i in range(population_size):
            if np.random.rand() < 0.03:  # 重新初始化部分个体
                positions[i] = np.random.uniform(lb, ub, dim)
            else:
                p = np.tanh(abs(fitness[i] - best_fitness))  # 位置更新概率
                vb = np.random.uniform(-a, a, dim)
                vc = np.random.uniform(-b, b, dim)

                # 记录选定个体的 vb 和 vc 值
                if i == random_individual_index:
                    vb_vc_history.append({
                        "Iteration": iteration + 1,
                        "vb_mean": np.mean(vb),
                        "vc_mean": np.mean(vc)
                    })

                for j in range(dim):
                    if np.random.rand() < p:  # 更新位置
                        A, B = np.random.choice(population_size, 2, replace=False)
                        positions[i, j] = best_position[j] + vb[j] * (
                                positions[A, j] - positions[B, j])
                    else:
                        positions[i, j] = vc[j] * positions[i, j]

        # 记录每次迭代的最佳适应度值
        convergence_curve.append(best_fitness)
        print(f"Iteration {iteration + 1}/{max_iterations}, Best Fitness: {best_fitness}")

    return best_position, best_fitness, convergence_curve, vb_vc_history
