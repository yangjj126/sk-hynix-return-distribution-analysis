# 6. 정규분포 곡선 만들기

# -0.08부터 0.06까지의 범위를
# 300개의 균등한 x좌표로 나누어 x라는 변수에 저장하시오.

import numpy as np
from scipy.stats import norm
# 이게 무슨 코드인데...
import matplotlib.pyplot as plt


x = np.linspace(
    -0.08,
    0.08,
    300
)


# 방금 네가 만든 x의 각 좌표에서
# 정규분포 곡선의 높이(y값)를 계산해서 y축 정보를 계산한다.
mu = 0.001
sigma = 0.02

normal_curve = norm.pdf(
    x,
    loc=mu,
    scale=sigma
)
# x - 전체에서 각각의 높이 제공
# loc - 평균의 위치 제공
# scale - 표준편차 제공


plt.plot(x, normal_curve)
# x를 가로축에 놓고, normal_curve를 y에 위치 시킨다.

plt.show()
# 진짜로 보여주는 코드