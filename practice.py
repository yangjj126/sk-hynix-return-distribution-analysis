import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
from scipy import stats

# sk하이닉스 주가 데이터 가져오기
ticker = yf.Ticker("000660.KS")

# 5년치 데이터 , 일별 주가, 보정해서
df = ticker.history(
    period="10y",
    interval="1d",
    auto_adjust=True
)

# 일단 5년치 데이터 출력해보셈
# print(df)

# 주가 데이터 갯수 구하기
# print("전체 주가 데이터 갯수:" , len(df))

# 종가만 가져오기
# print("Sk하이닉스 종가 데이터")
# print(df["Close"])

# 일별 수익률 데이터를 구한다
daily_data = df["Close"].pct_change(fill_method=None).dropna()
# daily_data_percentage = daily_data * 100
# first = daily_data_percentage.round().astype(int)
# second = first.astype(str) + "%"
# print(second)

print()
print("=== 일별 수익률 ===")
print(daily_data.head())

print()
print("전체 수익률 개수:", len(daily_data))

# ================
# 최근 일별 수익률 1000개 가져오기
# tail = 맨뒤순으로 데이터 1000개 들고 오기
# ================
returns_1000 = daily_data.tail(8000)
print(returns_1000)


# ==================================================
# 5. 평균과 표준편차 계산
# ==================================================
mu = returns_1000.mean()
sigma = returns_1000.std()

print()
print("=== 기본 통계 ===")

print("평균 일일 수익률:", mu)
print("평균 일일 수익률(%):", mu * 100)

print("수익률 표준편차:", sigma)

# ex = np.linspace(0,10,5) -> 0~10을 5구간으로 나누어라..
# print(ex)

# 6. 정규분포 곡선 만들기
x = np.linspace(
 returns_1000.min(),
 returns_1000.max(),
 500
)
# np.linespace(a, b, c) = a~b까지의 숫자를 c개의 점으로 나누느것
# a = returns_1000.min() : 1000개의 일별수익률 데이터중에서 최솟값
# b = returns_1000.max() : 1000개의 일별수익률 데이터중에서 최대값
# c = 500 : 500개의 점으로 나누는 것

normal_curve = norm.pdf(
    x,
    loc = mu,
    scale = sigma
)
# x에 들어있는 지점들 각각에서, 높이가 얼마인지를 계산해줘.
# mu - 중심위치, sigma - 퍼진 정도
# x - 각각의 위치
# norm.pdf(~) - x에 있는 모든 지점에 대해 높이를 다 계산




# norm - 정규분포(normal distribution)
# 평균과 표준편차를 알면 알 수 있다.
# mu = returns_1000.mean() - 전체수익률 1000개의 평균
# sigma = returns_1000.std() - 전체수익률 1000개의 표준편차

# loc은 location(위치)의 줄임말이야.
# 곡선의 정중앙을 x축 어디에 놓을지 정해.

# scale은 곡선의 폭을 결정해. 여기 들어가는 sigma(표준편차)는
# "수익률이 평균에서 얼마나 흩어져 있나"를 나타내는 숫자야.
# scale즉, sigma가 크면, 평균을 기준으로 많이 흩어져있다는 뜻
# sigma가 작으면, 평균을 기준으로 조금만 흩어져있다는 뜻



# 실제 수익률 히스토그램 + 정규분포 비교
plt.figure(figsize=(10,6))

# 실제 수익률 히스토그램 - 그리기
# hist는 막대그래프로 보여준다.
plt.hist(
    returns_1000,
    bins=50,
    density=True,
    alpha=0.6,
    label="Actual SK Hynix Returns"
)

# 정규분포 곡선 - 그리기
# 이미 계산해놓은 좌표를 점으로 찍고, 잇는다 - 정규분포곡선 완성
plt.plot(
    x,
    normal_curve,
    linewidth=2,
    label="Normal Distribution"
)


plt.axvline(
    mu,
    linestyle="--",
    label="Mean"
)
# 안중요


plt.title(
    "SK Hynix Daily Returns vs Normal Distribution"
)


plt.xlabel("Daily Return")
plt.ylabel("Density")

plt.legend()

plt.grid(alpha=0.3)

plt.show()


# 결과를 봣을때, 문제점
# 1. 정규분포보다 실제 데이터는 극단값들의 비중이 생각보다 크다
# 2. 중간부근에 분포하는 비중이 정규분포보다 실제 데이터는 그 비중이 더 크다


# 1번 수정사항
df_fit, loc_fit, scale_fit = t.fit(returns_1000)
# 정규분포가 안맞으니깐, 이제 t-분포로 바꿔보자
# t분포는 파라미터가 3개 (df, loc, scale)
# fit()이라는 함수를 통해서, 실제 막대그래프와 가장 유사한 형태로 만들수있는 
# 변수들을 설정을 한다.
# 그렇게 찾은 값으로 t.pdf를 돌리면, 실제 데이터를 잘 흉내낸, t-분포 곡선이 그려진다

print(f"추정된 자유도(df): {df_fit: .2f}")
print(f"추정된 중심(df): {loc_fit: .5f}")
print(f"추정된 중심(df): {scale_fit: .5f}")






# 아래는 다시 공부를 해야할 부분





# ==================================================
# 2번 수정사항 - t-분포 곡선 만들기
# ==================================================
t_curve = t.pdf(
    x,
    df = df_fit,
    loc = loc_fit,
    scale = scale_fit
)
# norm.pdf 때랑 똑같은 원리
# 차이점: norm 대신 t를 쓰고, df_fit이 추가됨 (꼬리 두께 반영)
# x는 재사용 (이미 만들어둔 500개 좌표 그대로)
# df_fit, loc_fit, scale_fit은 위에서 t.fit()으로 찾아낸 값


# ==================================================
# 히스토그램 + 정규분포 + t분포 - 셋 다 겹쳐서 그리기
# ==================================================
plt.figure(figsize=(10, 6))

# 실제 수익률 히스토그램
plt.hist(
    returns_1000,
    bins=50,
    density=True,
    alpha=0.6,
    label="Actual SK Hynix Returns"
)

# 정규분포 곡선 (기존)
plt.plot(
    x,
    normal_curve,
    linewidth=2,
    label="Normal Distribution",
    color="orange"
)

# t-분포 곡선 (새로 추가)
plt.plot(
    x,
    t_curve,
    linewidth=2,
    label="t-Distribution (Fat Tail)",
    color="green"
)

plt.axvline(
    mu,
    linestyle="--",
    label="Mean"
)

plt.title(
    "SK Hynix Daily Returns: Normal vs t-Distribution"
)

plt.xlabel("Daily Return")
plt.ylabel("Density")

plt.legend()

plt.grid(alpha=0.3)

plt.show()

# 이 그래프에서 확인할 것
# - 파란 막대(실제 데이터)의 뾰족한 봉우리 & 두꺼운 꼬리를
# - 초록 선(t-분포)이 주황 선(정규분포)보다 더 잘 따라가는지







# # ------- 몬테카를로 시뮬레이션 적용하기 ---------------
# n_days = 252
# n_sims = 100
# initial_price = df["Close"].iloc[-1]

# # n_days : 앞으로 몇일 동안 시뮬레이션을 돌릴지 - 1년치(252)
# # n_sims : 가능한 미래를 몇 개나 그려볼지 - 100개의 서로다른 시나리오를 무작위로 만들어볼거임
# # initial_price = df["Close"].iloc[-1] - df["Close"]라는 데이터 프레임에서 가장마지막행(가장 최근 가격)을 들고온다



# # 방법 A. 정규분포 기반 시뮬레이션
# Z_normal = np.random.normal(0, 1, size=(n_sims, n_days))
# # 평균0, 표준편차1 - 표준정규분포, size=(n_sims, n_days)
# # 일단, 대충 기본값으로 설정해놓은 것 - 순수한 무작위성

# daily_returns_normal = mu + sigma * Z_normal
# # - sigma를 곱하기 - 흩어진 정도가 SK하이닉스만큼 커진다
# # - mu를 더하기 - 중심이 sk하이닉스의 평균위치로 이동한다 (단, 1년짜리)


# price_paths_normal = initial_price * np.cumprod(1 + daily_returns_normal, axis=1)



# ==================================================
# 몬테카를로 시뮬레이션 (정규분포 vs t분포)
# ==================================================

n_days = 600
n_sims = 100
initial_price = df["Close"].iloc[-1]

# 방법 A: 정규분포 기반
Z_normal = np.random.normal(0, 1, size=(n_sims, n_days))
daily_returns_normal = mu + sigma * Z_normal
price_paths_normal = initial_price * np.cumprod(1 + daily_returns_normal, axis=1)








# 방법 B: t분포 기반 (fat tail 반영)
Z_t_raw = t.rvs(df=df_fit, size=(n_sims, n_days))
Z_t = Z_t_raw / np.sqrt(df_fit / (df_fit - 2))
daily_returns_t = loc_fit + scale_fit * Z_t
price_paths_t = initial_price * np.cumprod(1 + daily_returns_t, axis=1)

# 부채꼴 그래프로 비교
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

axes[0].plot(price_paths_normal.T, alpha=0.3, color="orange")
axes[0].set_title("Normal Distribution Simulation")
axes[0].set_xlabel("Days")
axes[0].set_ylabel("Price")

axes[1].plot(price_paths_t.T, alpha=0.3, color="green")
axes[1].set_title("t-Distribution Simulation (Fat Tail)")
axes[1].set_xlabel("Days")

plt.tight_layout()
plt.show()

# 1년 뒤 최종 가격 분포 비교
final_normal = price_paths_normal[:, -1]
final_t = price_paths_t[:, -1]

print()
print("=== 1년 뒤 예상 가격 비교 ===")
print(f"정규분포 - 평균: {final_normal.mean():,.0f}원, 최저: {final_normal.min():,.0f}원, 최고: {final_normal.max():,.0f}원")
print(f"t분포    - 평균: {final_t.mean():,.0f}원, 최저: {final_t.min():,.0f}원, 최고: {final_t.max():,.0f}원")

print()
print("=== 최악 시나리오 (하위 5%) 비교 ===")
print(f"정규분포 5백분위: {np.percentile(final_normal, 5):,.0f}원")
print(f"t분포 5백분위: {np.percentile(final_t, 5):,.0f}원")


print()
print("=== 1년 뒤 가격 범위 (t분포 기준) ===")
p5 = np.percentile(final_t, 5)
p25 = np.percentile(final_t, 25)
p50 = np.percentile(final_t, 50)
p75 = np.percentile(final_t, 75)
p95 = np.percentile(final_t, 95)

print(f"90% 확률로 이 범위 안: {p5:,.0f}원 ~ {p95:,.0f}원")
print(f"50% 확률로 이 범위 안: {p25:,.0f}원 ~ {p75:,.0f}원")
print(f"중앙값(가장 가능성 높은 지점): {p50:,.0f}원")