heights = [8, 8, 8]
heights.append(0)
stack = [-1]
ans = 0
for i in range(len(heights)):
    # 第一轮 heights = [....,0]
    while heights[i] < heights[stack[-1]]:
        h = heights[stack.pop()]
        w = i - stack[-1] - 1
        ans = max(ans, h * w)
    stack.append(i)
heights.pop()

