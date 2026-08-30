# 7. 정규분포와 히스토리(실제수익률) 비교하기
import matplotlib.pyplot as plt

returns_1000 = [0.01, 0.03, -0.04]

plt.figure(figsize=(10, 6))
# plt.figure() - 기본크기의 빈 도화지를 만든다.
# figsize: 그림의 크기

# 실제 Sk하이닉스 과거 일별수익률 데이터
plt.hist(
    returns_1000,
    # sk하이닉스 일별수익률 1000개
    bins=50,
    # 구간을 50개를 만든다.
    # SK하이닉스 수익률 전체 범위를 50개의 작은 구간으로 나누겠다.
    density=True,
    # 확률 밀도(density)로 표시를 한다 (normal_distribution 곡선 그래프와 맞추기 위해서)
    alpha=0.6,
    # 너무 진하면 겹쳤을때, 안보일수 있기 때문에, 약간 투명하게 만듦
    # 작아질수록 투명 (1~0)
    label="Actual Sk Hynix Returns"
    # 그래프 이름 붙이기 ("Actual SK Hynix Returns")
)

x = []
normal_curve = []

plt.plot(
    x,
    normal_curve,
    linewidth=2,
    label="Normal Distribution"
)




