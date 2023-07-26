# boka = 1  # бока ходит первым
# jora = 0
# n, m = list(map(int, input().split(' ')))
# n_data = list(map(int, input().split(' ')))
# m = 3

# n = 4
# n_data = [1, 2, 3, 9]
# n_data = [1, 2, 3, 9]
# 1 1 1  100000 это похоже на задачу про то, как поставить + и минусы
# чтобы в итоге получить максимальный эффект
# Sj = n_data[0]
# i = 1
# while i <= m:
#     if n_data[i] < n_data[i + m]:
#         Sj += n_data[i]
#         i += 1
#     else:
#         pass
# i =

# if n <= m + 1:
#     if n_data[1] > n_data[1] + n_data[2]
# i = 1
# Sj = n_data[0]
# j = 1
# k = []
# while i < len(n_data):
#     # print(i, j, Sj, m)
#     if i + m <= len(n_data):
#         if n_data[i] > n_data[i + m - 1]:
#             Sj = Sj + j * n_data[i]
#         else:
#             j = -1
#             Sj = Sj + j * n_data[i]
#         i += 1
#     else:
#         m -= 1
#
# # print(Sj)
# if Sj > 0:
#     print(1)
# else:
#     print(0)
# s = 0
# for i in range(1, 51):
#     s += i*(i-1) - (50-i)*3
#     print(i)
# print(s/50)
# Найти Е(суммарно набранное число очков) =
# Е(сумма по игрокам, у которых в паре выпало одинак колич) -
# Е(сумма по всем 50 мгрокам, где каждый элемент суммы это квадрат количесва людей с той же гранью) =
# для первого ожтдания:
# количество пар с 1

# a = [1, 1]
# 1 + 1 -4 - 4 = -6
# для каждого броска -
# считаем, сколько с ним пар и прибавляем значение при броске умножить на количество пар
# вычитаем квадрат числа людей с таким же числом при броске.


# def culc_i(N):
#     freq = list(range(1, N + 1))
#     s = 0
#     for fr in freq:  # все варианты количества того, что может выпасть
#         q_par = fr - 1
#         minus =  fr ** 2
#         for ii in range(1, N + 1): # все варианты того, что может выпасть
#             plus =  q_par * ii
#             s += plus - minus
#             print(plus - minus)
#     return s

# print(culc_N(2))
# print(sum(list(range(1, 51)))/50)
# def calc_sum(a):
#     s = 0
#     for i in a:
#         for qlch in freq:
#             q_par = qlch - 1
#
#
#         freq = a.count(i)
#         q_par = freq - 1
#         s += q_par * i
#         s -= freq ** 2
#     return s
#
# def calc_exp(N):
#     data = N * [1]
#     result = []
#     i = 0
#     j = 1
#     while i < N:
#         while j <= N:
#             result.append(calc_sum(data))
#             data[i] = j
#             print(data)
#             j += 1
#         j = 1
#         i += 1
#     return result, sum(result)/len(result), len(result)
#
# print(calc_exp(4))
#
#
#
# def culc_N(N):
#     freq = list(range(1, N + 1))
#     s = 0
#     for i in range(N): # для каждого броска подсчитаем
#         for fr in freq:  # все варианты количества того, что может выпасть
#             q_par = fr - 1
#             minus =  fr ** 2
#             for ii in range(1, N + 1): # все варианты того, что может выпасть
#                 plus =  q_par * ii
#                 s += plus - minus
#                 print(plus - minus)
#     return s
import math
def calc_sum(a): #[1,1,2,2]
    s = 0
    for i in set(a):
        freq = a.count(i)
        s += freq * (freq - 1) * i - freq ** 3
    return s

# print(calc_sum([1, 2]))
# print(890)
#
# import numpy as np
# r = []
# l = 50
# for it in range(100000):
#     a = np.random.randint(l, size=(l))
#     if it % 10000 == 0:
#         print(it/100000)
#     r.append(calc_sum(list(a + 1)))
# print(a + 1) -5.66566  -5.285714285714286 Для 8 это -2.75, для 7 это -5.28  для 6 это -6.833333333333333
# a = len(a[a == 1])/len(a) а для 8 в методе монте-карло - это
# print(a)
# ln = len(r)
# print(sum(r)/ln)
# jl = 0
#
# for i in set(r):
#     weight = r.count(i) /ln
#     jl += weight * i
# print(max(set(r), key=r.count))
# print(jl)

import itertools

# print(list(itertools.permutations([1, 2, 3])))


# for p in next_permutation([int(c) for c in "111222"]):
#     print(p) -5.285714285714286
# import itertools
# val = list(range(1, 8))
# com_set = itertools.product(val,repeat=7)
# jf = []
# for i in com_set:
#     jf.append(calc_sum(i))
#     print(i)
# print(sum(jf)/len(jf))
#
# print(len(jf), 5 ** 5)
