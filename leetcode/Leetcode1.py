# 开始使用python完成LeetCode上的算法题

# 78. Subsets
# Given a set of distinct integers, nums, return all possible subsets (the power set).
# Note: The solution set must not contain duplicate subsets.
from typing import List


class Solution:
    # def subsets(self, nums: List[int]) -> List[List[int]]:
    def subsets1(self, nums):
        res = []
        self.dfs(sorted(nums), 0, [], res)
        return res

    def dfs(self, nums, index, path, res):
        res.append(path)
        for i in range(index, len(nums)):
            self.dfs(nums, i + 1, path + [nums[i]], res)

    def subset2(self, nums):
        res = [[]]
        for num in sorted(nums):
            res += [item + [num] for item in res]
        return res

    #def exist(self, board: List[List[str]], word: str) -> bool:



if __name__ == "__main__":
    Solution.subsets1()


