"""
file: clever.py
author: 汪子策
date: 2025-4-14
version: 1.0
license: ARR
Copyright (c) 2025 汪子策. All rights reserved.
"""
from itertools import product

# 遍历所有可能的答案组合
for cob in product('ABCDE', repeat=10):
    # 将元组转换为列表，方便索引
    ans = list(cob)

    # 检查问题1
    if ans[0] == 'B':
        if ans[2] != 'C':
            continue
    elif ans[0] == 'C':
        if ans[3] != 'B':
            continue
    elif ans[0] == 'D':
        if ans[4] != 'B':
            continue
    elif ans[0] == 'E':
        if ans[5] != 'B':
            continue

    # 检查问题2
    hs = False
    if ans[1] == 'A':
        hs = ans[1] == ans[2]
    elif ans[1] == 'B':
        hs = ans[2] == ans[3]
    elif ans[1] == 'C':
        hs = ans[3] == ans[4]
    elif ans[1] == 'D':
        hs = ans[4] == ans[5]
    elif ans[1] == 'E':
        hs = ans[5] == ans[6]

    if not hs:
        continue

    # 检查问题3
    if ans[2] == 'A':
        if ans[2] != ans[0]:
            continue
    elif ans[2] == 'B':
        if ans[2] != ans[1]:
            continue
    elif ans[2] == 'C':
        if ans[2] != ans[3]:
            continue
    elif ans[2] == 'D':
        if ans[2] != ans[6]:
            continue
    elif ans[2] == 'E':
        if ans[2] != ans[5]:
            continue

    # 检查问题4
    ca = ans.count('A')
    if ans[3] == 'A' and ca != 0:
        continue
    elif ans[3] == 'B' and ca != 1:
        continue
    elif ans[3] == 'C' and ca != 2:
        continue
    elif ans[3] == 'D' and ca != 3:
        continue
    elif ans[3] == 'E' and ca != 4:
        continue

    # 检查问题5
    if ans[4] == 'A':
        if ans[4] != ans[9]:
            continue
    elif ans[4] == 'B':
        if ans[4] != ans[8]:
            continue
    elif ans[4] == 'C':
        if ans[4] != ans[7]:
            continue
    elif ans[4] == 'D':
        if ans[4] != ans[6]:
            continue
    elif ans[4] == 'E':
        if ans[4] != ans[5]:
            continue

    # 检查问题6
    cb = ans.count('B')
    c_c = ans.count('C')
    c_d = ans.count('D')
    c_e = ans.count('E')

    if ans[5] == 'A' and ca != cb:
        continue
    elif ans[5] == 'B' and ca != c_c:
        continue
    elif ans[5] == 'C' and ca != c_d:
        continue
    elif ans[5] == 'D' and ca != c_e:
        continue
    elif ans[5] == 'E' and ca == cb or ca == c_c or ca == c_d or ca == c_e:
        continue

    # 检查问题7
    if ans[6] == 'A':
        diff = abs(ord(ans[6]) - ord(ans[7]))
        if diff != 4:
            continue
    elif ans[6] == 'B':
        diff = abs(ord(ans[6]) - ord(ans[7]))
        if diff != 3:
            continue
    elif ans[6] == 'C':
        diff = abs(ord(ans[6]) - ord(ans[7]))
        if diff != 2:
            continue
    elif ans[6] == 'D':
        diff = abs(ord(ans[6]) - ord(ans[7]))
        if diff != 1:
            continue
    elif ans[6] == 'E':
        diff = abs(ord(ans[6]) - ord(ans[7]))
        if diff != 0:
            continue

    # 检查问题8
    cv = ans.count('A') + ans.count('E')
    if ans[7] == 'A' and cv != 2:
        continue
    elif ans[7] == 'B' and cv != 3:
        continue
    elif ans[7] == 'C' and cv != 4:
        continue
    elif ans[7] == 'D' and cv != 5:
        continue
    elif ans[7] == 'E' and cv != 6:
        continue

    # 检查问题9
    cc = 10 - cv
    if ans[8] == 'A':
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n ** 0.5) + 1):
                if n % i == 0:
                    return False
            return True
        if not is_prime(cc):
            continue
    elif ans[8] == 'B':
        if cc not in [1, 1, 2, 6, 24, 120]:
            continue
    elif ans[8] == 'C':
        squares = [i ** 2 for i in range(11)]
        if cc not in squares:
            continue
    elif ans[8] == 'D':
        cubes = [i ** 3 for i in range(5)]
        if cc not in cubes:
            continue
    elif ans[8] == 'E':
        if cc % 5 != 0:
            continue

    # 如果所有条件都满足，则输出答案
    print(" ".join(ans))