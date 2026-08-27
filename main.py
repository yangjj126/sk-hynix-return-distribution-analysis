# import yfinance as yf

# ticker = yf.Ticker("000660.KS")

# df = ticker.history(
#     period="5y",
#     interval="1d",
#     auto_adjust=True
#     # auto_adjust = True로 설정을 하면
#     # 액면분할을 하거나, 배당 지급이후 가격하락
#     # 을 주가로 기록하면 안되기 때문에, auto_adjust를 True로 설정해야 한다
# )


# # Close 열만 선택
# close = df["Close"]
# print(close)

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats


# ==================================================
# 1. SK하이닉스 주가 데이터 가져오기
# ==================================================

ticker = yf.Ticker("000660.KS")

df = ticker.history(
    period="5y",
    interval="1d",
    auto_adjust=True
)

print("===== SK하이닉스 주가 데이터 =====")
print(df.head())

print()
print("전체 주가 데이터 개수:", len(df))


# ==================================================
# 2. 종가(Close)만 가져오기
# ==================================================

close = df["Close"]

print()
print("===== 종가 데이터 =====")
print(close.head())


# ==================================================
# 3. 일별 수익률 계산
# ==================================================

returns = close.pct_change(fill_method=None).dropna()
# 1. pct_change = 오늘 수익률(오늘이 어제 대비 몇% 변했는가) - Percent Change
# 2. fill_method=None : 중간에 결측치인 값에 대해서는 값을 채우는 것이 아닌, 그대로 둔다
# 3. dropna() : 첫번째데이터를 제거하는 것이다.

print()
print("===== 일별 수익률 =====")
print(returns.head())

print()
print("전체 수익률 개수:", len(returns))


# ==================================================
# 4. 최근 일별 수익률 1000개 가져오기
# ==================================================

returns_1000 = returns.tail(1000)

print()
print("===== 최근 1000개 수익률 =====")
print(returns_1000)

print()
print("사용한 수익률 개수:", len(returns_1000))


# ==================================================
# 5. 평균과 표준편차 계산
# ==================================================

mu = returns_1000.mean()
sigma = returns_1000.std()
# sigma = volatility(변동성) - 수익률이 얼마나 흩어져있는지
# ex) 0.01 - 하루 등락폭 +,- 1% 안팎, 0.03 - 하루 등락폭 +,- 3%
# std = standard deviation(표준편차)


print()
print("===== 기본 통계 =====")

print("평균 일일 수익률:", mu)
print("평균 일일 수익률(%):", mu * 100)

print()

print("일일 표준편차:", sigma)
print("일일 표준편차(%):", sigma * 100)



# ==================================================
# 6. 정규분포 곡선 만들기
# ==================================================

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
    loc=mu,
    scale=sigma
)
# norm - 정규분포(normal distribution)
# 평균과 표준편차를 알면 알 수 있다.
# 


# ==================================================
# 7. 실제 수익률 히스토그램 + 정규분포 비교
# ==================================================

plt.figure(figsize=(10, 6))
# 이미지(정규분포)의 크기를 설정해주는 것..

plt.hist(
    returns_1000,
    bins=50,
    # 구간갯수: 50개
    density=True,
    # -> 갯수가 아니라, 확률밀도로 표시, - 정규분포곡선단위(확률)이기 때문에
    alpha=0.6,
    # 투명도 - 뒤에 겹쳐질 "정규분포곡선"이 막대에 가려지지 않고 비치게 하기 위해서
    label="Actual SK Hynix Returns"
    # label - 이름표
)
# 히스토그램 - 데이터를 구간별로 몇개씩 있는지에 대한 데이터를 시각화한것


plt.plot(
    x,
    normal_curve,
    linewidth=2,
    label="Normal Distribution"
)
# 정규분포 곡선 그리기




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


# ==================================================
# 8. Q-Q Plot
# ==================================================
# 내 실제 데이터가 정규 분포를 따를 경우에, 있어야 할 자리에 
# 진짜로 있는지 판단을 해주는 코드임...
plt.figure(figsize=(7, 7))

stats.probplot(
    returns_1000,
    dist="norm",
    plot=plt
)

plt.title(
    "Q-Q Plot of SK Hynix Daily Returns"
)

plt.grid(alpha=0.3)

plt.show()


# ==================================================
# 9. Shapiro-Wilk 정규성 검정
# ==================================================

shapiro_stat, p_value = stats.shapiro(returns_1000)

print()
print("===== Shapiro-Wilk 정규성 검정 =====")

print("검정통계량:", shapiro_stat)
print("p-value:", p_value)

if p_value > 0.05:
    print("결론: 정규분포가 아니라고 판단할 충분한 증거가 없습니다.")
else:
    print("결론: 정규분포를 따른다는 가설을 기각합니다.")


# ==================================================
# 10. 왜도와 첨도 확인
# ==================================================

skewness = stats.skew(returns_1000)
kurtosis = stats.kurtosis(returns_1000)

print()
print("===== 분포 모양 =====")

print("왜도 (Skewness):", skewness)
print("초과첨도 (Excess Kurtosis):", kurtosis)


# ==================================================
# 몬테카를로 시뮬레이션 (정규분포 vs t분포)
# ==================================================

n_days = 252
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