"""蒙特卡洛树搜索 (MCTS),AlphaZero 变体。

使用 PUCT 选择公式,由神经网络评估叶节点,无 rollout。

性能优化:
  - 批量推理 (virtual loss):单次搜索并发收集多个叶节点凑成一批,一次性送
    网络评估,大幅减少 GPU 调用次数。virtual loss 在选择时临时给在途路径加
    "虚拟失败",避免一批内的并发模拟都挤向同一条路径。
  - 树复用:走子后把对应子节点提升为新根,保留其子树统计(见 advance_root)。

价值约定:网络输出与节点存储的价值均为"该节点轮走方"视角。回传时按层取反,
使每个节点累积的是自己视角的价值。
"""

from __future__ import annotations

import math

from .game import GameState

VIRTUAL_LOSS = 1.0  # 每条在途路径施加的虚拟失败量


class Node:
    __slots__ = ("prior", "to_move", "visit_count", "value_sum",
                 "children", "is_expanded")

    def __init__(self, prior: float, to_move: int):
        self.prior = prior            # 父节点策略给出的先验 P(s,a)
        self.to_move = to_move        # 该节点局面轮到谁走
        self.visit_count = 0          # N(s,a)
        self.value_sum = 0.0          # W(s,a) 累积价值(本节点视角)
        self.children: dict = {}      # move -> Node
        self.is_expanded = False

    def value(self) -> float:
        """Q(s,a) = W/N。"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, evaluator, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25,
                 batch_size: int = 8):
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.batch_size = batch_size

    def run(self, state: GameState, num_simulations: int,
            add_noise: bool = True, rng=None, root: Node | None = None):
        """对给定根局面运行搜索,返回根节点(含访问统计)。

        root 可传入复用的子树根;若其尚未扩展则先评估扩展。
        """
        if root is None:
            root = Node(prior=1.0, to_move=state.to_move)
        if not root.is_expanded:
            self._expand(root, state)
        if add_noise:
            self._add_dirichlet_noise(root, rng)

        sims_done = 0
        while sims_done < num_simulations:
            sims_done += self._run_batch(
                root, state, num_simulations - sims_done)
        return root

    def _run_batch(self, root: Node, root_state: GameState, remaining: int) -> int:
        """收集一批叶节点并批量评估。返回本批完成的模拟数。

        每次模拟从根选择到叶,沿途施加 virtual loss(使后续模拟倾向其他路径);
        终局叶直接定值回传,非终局叶暂存,凑批后一次性送网络评估再扩展回传。
        """
        batch = min(self.batch_size, remaining)
        pending = []  # [(leaf_node, leaf_state, path)]
        completed = 0

        for _ in range(batch):
            node = root
            state = root_state.copy()
            path = [node]

            while node.is_expanded and node.children:
                move, node = self._select_child(node)
                state.push(move)
                path.append(node)
                self._apply_virtual_loss(node)

            result = state.result()
            if result != "ongoing":
                value = self._terminal_value(result, state.to_move)
                self._backpropagate(path, value, revert_virtual=True)
                completed += 1
            else:
                pending.append((node, state, path))

        if pending:
            states = [p[1] for p in pending]
            evals = self.evaluator.evaluate_batch(states)
            for (node, state, path), (policy, value) in zip(pending, evals):
                self._expand_with_policy(node, state, policy)
                self._backpropagate(path, value, revert_virtual=True)
                completed += 1

        return completed

    @staticmethod
    def _apply_virtual_loss(node: Node):
        """给在途节点施加 virtual loss:增计一次访问并累计一个失败价值。"""
        node.visit_count += 1
        node.value_sum -= VIRTUAL_LOSS

    def _select_child(self, node: Node):
        """用 PUCT 选择得分最高的子节点。"""
        total_visits = sum(c.visit_count for c in node.children.values())
        sqrt_total = math.sqrt(total_visits + 1)

        best_score = -float("inf")
        best_move = None
        best_child = None
        for move, child in node.children.items():
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            # 子节点 Q 是子节点(对方)视角,对当前节点而言取负。
            q = -child.value()
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child
        return best_move, best_child

    def _expand(self, node: Node, state: GameState):
        """评估并扩展节点(用于根节点初始化)。"""
        result = state.result()
        if result != "ongoing":
            node.is_expanded = True
            return
        policy, _ = self.evaluator.evaluate(state)
        self._expand_with_policy(node, state, policy)

    def _expand_with_policy(self, node: Node, state: GameState, policy: dict):
        # 同一节点可能在一批内被多条路径选为叶,只扩展一次。
        if node.is_expanded:
            return
        next_to_move = -state.to_move
        for move, prior in policy.items():
            node.children[move] = Node(prior=prior, to_move=next_to_move)
        node.is_expanded = True

    def _backpropagate(self, path: list, leaf_value: float,
                       revert_virtual: bool):
        """从叶到根回传。leaf_value 为叶节点轮走方视角的价值。

        若选择阶段施加过 virtual loss(根节点除外,根不加 VL),回传时先撤销
        虚拟量再计入真实统计。撤销:visit-1, value_sum+=VL;真实:visit+1,
        value_sum+=value。合并后:visit 不变(已在 VL 时 +1),value_sum
        += VL + value。
        """
        value = leaf_value
        for i, node in enumerate(reversed(path)):
            is_root = (i == len(path) - 1)
            if is_root:
                # 根节点未施加 virtual loss,正常计入。
                node.visit_count += 1
                node.value_sum += value
            elif revert_virtual:
                # 撤销 virtual loss 并计入真实价值;visit 已在 VL 时 +1。
                node.value_sum += VIRTUAL_LOSS + value
            else:
                node.visit_count += 1
                node.value_sum += value
            value = -value  # 上一层是对方视角

    @staticmethod
    def _terminal_value(result: str, to_move: int) -> float:
        """终局对 to_move 一方的价值:胜 +1,负 -1,和 0。"""
        from .constants import RED
        if result == "draw":
            return 0.0
        red_won = result == "red_win"
        to_move_is_red = to_move == RED
        if red_won == to_move_is_red:
            return 1.0
        return -1.0

    def _add_dirichlet_noise(self, root: Node, rng):
        """在根节点先验上混入 Dirichlet 噪声,保证自我对弈的探索。"""
        if not root.children:
            return
        moves = list(root.children.keys())
        if rng is not None:
            noise = rng.dirichlet([self.dirichlet_alpha] * len(moves))
        else:
            import numpy as np
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        eps = self.dirichlet_eps
        for move, n in zip(moves, noise):
            child = root.children[move]
            child.prior = (1 - eps) * child.prior + eps * float(n)


def advance_root(root: Node, move) -> Node | None:
    """走子后复用子树:返回 move 对应的子节点作为新根,找不到返回 None。

    新根的 prior 不再有意义(它将作为根重新评估/加噪),但其 children 与访问
    统计得以保留,省去重复搜索。
    """
    if root is None or not root.children:
        return None
    return root.children.get(tuple(move))


def action_probabilities(root: Node, temperature: float = 1.0):
    """由根节点访问次数导出走子概率分布 π。

    temperature=1: 正比于访问次数;temperature->0: 趋向 one-hot(选最多访问)。
    返回 dict[move -> prob]。
    """
    moves = list(root.children.keys())
    visits = [root.children[m].visit_count for m in moves]

    if temperature <= 1e-2:
        best = max(range(len(moves)), key=lambda i: visits[i])
        probs = [0.0] * len(moves)
        probs[best] = 1.0
        return dict(zip(moves, probs))

    # π(a) ∝ N(a)^(1/τ)。先按最大访问数归一化再做幂,避免大指数溢出
    # (公共因子在最终归一化时约掉,结果与直接幂等价)。
    max_visit = max(visits)
    if max_visit == 0:
        u = 1.0 / len(moves)
        return {m: u for m in moves}
    inv_t = 1.0 / temperature
    powed = [(v / max_visit) ** inv_t for v in visits]
    total = sum(powed)
    return {m: p / total for m, p in zip(moves, powed)}



