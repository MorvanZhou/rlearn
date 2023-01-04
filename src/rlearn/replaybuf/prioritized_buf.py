import typing as tp

import numpy as np

from rlearn.replaybuf.base import BaseReplayBuffer


class SumTree:
    def __init__(self, max_size):
        self.max_size = max_size
        self._tree_body_size = 2 * max_size - 1
        self._pointer = 0
        self._is_full = False
        self._min_idx = 0
        self._max_idx = 0
        self.nodes = np.zeros(self._tree_body_size)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity

    def _renew_min_max_p_index(self, p, index):
        if p > self.max_p:
            self._max_idx = index
        if p < self.min_p:
            self._min_idx = index

    def add(self):
        pointer = self._pointer
        if pointer == 0 and not self.is_full():
            p = 1
        else:
            p = self.max_p
        self._renew_min_max_p_index(p, pointer)
        self.update(self._pointer, p)  # update tree_frame
        self._pointer += 1
        if self._pointer >= self.max_size:
            self._pointer = 0
            self._is_full = True
        return pointer

    def update(self, idx: int, p: float):
        """
        Args:
            idx (int): 叶子节点的 index
            p (float): priority 优先度
        """
        self._renew_min_max_p_index(p, idx)
        node_idx = idx + self.max_size - 1
        p_diff = p - self.nodes[node_idx]
        self.nodes[node_idx] = p
        # then propagate the change through tree
        while node_idx != 0:  # this method is faster than the recursive loop in the reference code
            node_idx = (node_idx - 1) // 2
            self.nodes[node_idx] += p_diff

    def get_index(self, p: float) -> int:
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]

        Args:
            p (float): 优先度

        Return:
            int: 数据的 index
        """
        parent_idx = 0
        while True:  # the while loop is faster than recursive method
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= self._tree_body_size:  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if p <= self.nodes[cl_idx]:
                    parent_idx = cl_idx
                else:  # if p is greater then node's value, subtract p from node's value and check right node
                    p -= self.nodes[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.max_size + 1
        return data_idx

    def is_full(self):
        return self._is_full

    @property
    def sum(self):
        return self.nodes[0]

    @property
    def bottom_nodes(self):
        bottom = self.nodes[-self.max_size:]
        if not self.is_full():
            return bottom[:self._pointer]
        return bottom

    @property
    def max_p(self):
        if len(self.bottom_nodes) == 0:
            return 1
        return self.bottom_nodes[self._max_idx]

    @property
    def min_p(self):
        if len(self.bottom_nodes) == 0:
            return 1
        return self.bottom_nodes[self._min_idx]

    @property
    def pointer(self):
        return self._pointer


class PrioritizedReplayBuffer(BaseReplayBuffer):
    name = __qualname__

    def __init__(self, max_size):
        super().__init__(max_size=max_size)
        self.epsilon = 0.01  # small amount to avoid zero priority
        self.alpha = 0.6  # [0~1] convert the importance of TD error to priority
        self.beta = 0.4  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = 0.001
        self.upper_abs_err_bound = 1.  # clipped abs error

        self.tree = SumTree(max_size=max_size)
        self._cache_sample_indices = []
        self._cache_sample_importance_sampling_weights = []

    def sample(self, batch_size: int):
        if self.tree.pointer == 0 and not self.is_full():
            raise ValueError("replay buffer is empty")
        segment_value = self.tree.sum / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = self.tree.min_p / self.tree.sum  # for later calculate IS_weights
        self._cache_sample_indices.clear()
        self._cache_sample_importance_sampling_weights.clear()
        for i in range(batch_size):
            lower, upper = segment_value * i, segment_value * (i + 1)
            random_priority = np.random.uniform(lower, upper)
            idx = self.tree.get_index(random_priority)
            idx_priority = self.tree.bottom_nodes[idx]
            prob_j = idx_priority / self.tree.sum
            self._cache_sample_importance_sampling_weights.append(np.power(prob_j / min_prob, -self.beta))
            self._cache_sample_indices.append(idx)

        ba = self.a[self._cache_sample_indices]
        br = self.r[self._cache_sample_indices]
        bs = self.s[self._cache_sample_indices]

        # next state
        if self.has_next_state:
            bs_ = bs[:, 1]
            bs = bs[:, 0]
            batch = (bs, ba, br, bs_)
        else:
            # no next state
            batch = (bs, ba, br, None)
        return batch

    def put_batch(
            self,
            s: np.ndarray,
            a: np.ndarray,
            r: np.ndarray,
            s_: tp.Optional[np.ndarray] = None
    ):
        states, a, r = self.preprocess_batch_data(s, a, r, s_)

        batch_size = a.shape[0]
        for i in range(batch_size):
            pointer = self.tree.add()
            self.s[pointer] = states[i]
            self.a[pointer] = a[i]
            self.r[pointer] = r[i]

        if self._is_empty and batch_size > 0:
            self._is_empty = False

        if self.tree.is_full():
            self._is_full = True

    def clear(self):
        self._is_full = False
        self._is_empty = True
        self.tree = SumTree(max_size=self.max_size)

    def batch_update(self, abs_errors: np.ndarray):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.upper_abs_err_bound)
        # how much prioritization is used, alpha=0 corresponding to uniform case
        ps = np.power(clipped_errors, self.alpha)
        for i, p in zip(self._cache_sample_indices, ps):
            self.tree.update(i, p)
        self._cache_sample_indices.clear()
        self._cache_sample_importance_sampling_weights.clear()

    @property
    def current_loading_point(self):
        if self.is_full():
            return self.max_size
        return self.tree.pointer

    @property
    def cache_importance_sampling_weights(self):
        return np.array(self._cache_sample_importance_sampling_weights, dtype=np.float32)
