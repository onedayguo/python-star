# 开始使用python完成LeetCode上的算法题
# --coding:utf-8

from typing import List
from leetcode import ListNode



class Solution:
    # 78. Subsets
    # Given a set of distinct integers, nums, return all possible subsets (the power set).
    # Note: The solution set must not contain duplicate subsets.
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

    # 79.二维字符数组中，上下左右寻找目标字符串
    def exist(self, board, word):
        if not board:
            return False
        for i in range(len(board)):
            for j in range(len(board[0])):
                if self.dfs(board, i, j, word):
                    return True
        return False

        # check whether can find word, start at (i,j) position

    def dfs(self, board, i, j, word):
        if len(word) == 0:  # all the characters are checked
            return True
        if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or word[0] != board[i][j]:
            return False
        tmp = board[i][j]  # first character is found, check the remaining part
        board[i][j] = "#"  # avoid visit agian
        # check whether can find "word" along one direction
        res = self.dfs(board, i + 1, j, word[1:]) or self.dfs(board, i - 1, j, word[1:]) \
        or self.dfs(board, i, j + 1, word[1:]) or self.dfs(board, i, j - 1, word[1:])
        board[i][j] = tmp
        return res

    # 80 去除重复字符超过两个的字符，保留两个重复字符，返回字符长度
    def removeDuplicates(self, nums: List[int]) -> int:
        # 双指针思想，跳过多余重复字符，重排列数组
        if len(nums) < 3:
            return len(nums)
        pos = 1
        for i in range(1, len(nums)-1):
            if nums[i-1] != nums[i+1]:
                nums[pos] = nums[i]
                pos += 1
        nums[pos] = nums[-1]
        return pos+1

    def removeDuplicates2(self, nums:List[int]) -> int:
        # 统计每个字符的数量，同一字符多于2个的去除
        for val in set(nums):
            while nums.count(val) > 2:
                nums.remove(val)
        return len(nums)

    # 33.Search in Rotated Sorted Array,二分查找
    def search(self, nums: List[int], target: int) -> int:
        low, high = 0, len(nums) - 1
        while low < high:
            mid = (low+high) // 2
            # nums[0] <= target <= nums[i]
            #            target <= nums[i] < nums[0]
            #                      nums[i] < nums[0] <= target
            if (nums[0] > target) ^ (nums[0] > nums[mid]) ^ (target > nums[mid]):
                low = mid+1
            else:
                high = mid
        return low if target in nums[low:low+1] else -1

    # 81. Search in Rotated Sorted Array II，二分查找
    def search81(self, nums: List[int], target: int) -> bool:
        """左中右 3个指针，把数组当做有序数组
            二分查找
        """
        left, right = 0, len(nums)-1
        while left <= right:
            mid = (right+left) // 2
            if nums[mid] == target:
                return True
            # 有序数组，左指针=中间值，则让左指针后移
            while left < mid and nums[left] == nums[mid]:  # tricky part
                left += 1
            # 前半段有序
            if nums[left] <= nums[mid]:
                # 目标值在前半段
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            # 后半段有序
            else:
                # 目标值在后半段
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return False

    # 82 链表去重，返回无重复节点链表
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        # 虚拟头结点作为返回节点，pre节点指向结果链表的末位节点
        dummy = pre = ListNode(0)
        dummy.next = head
        # 当前节点和后节点不空
        while head and head.next:
            # 发现重复值，while循环判断直到不是重复值
            if head.val == head.next.val:
                while head and head.next and head.val == head.next.val:
                    head = head.next
                head = head.next
                pre.next = head
            # 不重复，当前指针后移
            else:
                pre = pre.next
                head = head.next
        return dummy.next

    # 83.链表去重，保留单个重复节点
    def deleteDuplicates1(self, head: ListNode) -> ListNode:
        # 虚拟头结点作为返回节点，pre节点指向结果链表的末位节点
        dummy = pre = ListNode(0)
        # [1],考虑单个节点情况
        dummy.next = head
        # 当前节点和后节点不空
        while head and head.next:
            # 发现重复值，while循环判断直到不是重复值
            if head.val == head.next.val:
                while head and head.next and head.val == head.next.val:
                    head = head.next
                # head = head.next
                pre.next = head
            # 不重复，当前指针后移
            else:
                pre = pre.next
                head = head.next
        return dummy.next