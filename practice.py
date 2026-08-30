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
returns_1000 = daily_data.tail(1000)
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
# x = np.linespace(
#     returns_1000.min(),
#     returns_1000.max(),
#     500
# )
# # 1000개의 수익률 분포중, 최대, 최소 x좌표에 나타내고, 
# # 500개로 구간을 나누겠다.

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

# 여기까지가 정규분포 곡선 만드는 코드













# 7. 실제 수익률 히스토그램 + 정규분포 비교
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
# fit()이라는 함수를 통해서,
# sk하이닉스 일별수익률 1000개를 가장 그럴듯 하게 만들어내는 
# t-분포의 df, loc, scale을 찾는다..






print(f"추정된 자유도(df): {df_fit: .2f}")
print(f"추정된 중심(df): {loc_fit: .5f}")
print(f"추정된 중심(df): {scale_fit: .5f}")






# 아래는 다시 공부를 해야할 부분


# sk하이닉스 실제 일별수익률에 정규분포와 t분포 동시에 겹치기
# 어느 분포가 실제 데이터를 더 잘 설명하는지를 눈으로 비교하는 것

# ==================================================
# 2번 수정사항 - t-분포 곡선 만들기
# ==================================================
t_curve = t.pdf(
    x,
    df = df_fit,
    # df만 새로 추가 된것....
    loc = loc_fit,
    scale = scale_fit
)
# norm.pdf 때랑 똑같은 원리
# 가장 최적화된 loc(뮤), scale(표준편차), + 자유도(df)
# df는 꼬리의 두께와 전체 모양을 조절한다.
# df증가 -> 꼬리가 얇아진다, df감소 -> 꼬리가 두꺼워진다


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
# axvline = 그래프에 세로선을 하나 긋는것



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



























# ----------------------------------------------------------------------------------
# chat-gpt로 공부


# 결론적으로 t분포를 통해서 몬테카를로 시뮬레이션을 적용하는 것이
# 가장 적합한 방식이라고 판단을 했기 때문에
# t-분포를 토대로 몬테카를로 시뮬레이션을 돌릴것이다.



# 몬테카를로 시뮬레이션이란 무엇인가...
# example)
# - 시나리오 10000개를 돌린다는 것 = t분포에서 무작위 21개의 일별수익률
# 굳이 실제 데이터를 토대로 10000개의 시뮬레이션을 돌리지 않는 이유
# 과거에 실제로 나타나지 않은 수익률또한 시뮬레이션에 반영을 하고 싶기
# 때문에, 실제 데이터를 토대로 시뮬레이션을 돌리는 것이 아니라,
# t-분포를 토대로 시뮬레이션을 돌리는 것이다.
# -- 솔직히 t-분포도 좀 애매한데, 그럼에도 가정을 하는 것
# "미래 수익률이 우리가 추정한 t분포와 비슷할 것이다"
# -> "t-분포 Monte Carlo"





# ==================================================
# t-분포 기반 Monte Carlo Simulation
# ==================================================

# 앞으로 몇 거래일을 시뮬레이션할 것인지
n_days = 252

# 252거래일 ≈ 주식시장 기준 약 1년
#
# 21  → 약 1개월
# 63  → 약 3개월
# 252 → 약 1년


# ==================================================
# 가상의 미래를 몇 번 만들어볼 것인지
# ==================================================

n_sims = 10000

# 10000이라는 뜻:
#
# "앞으로 252일 동안 SK하이닉스 주가가 움직이는 미래"
# 를 10000개 만들어본다는 뜻
#
# 시나리오 1 → 가상의 252일
# 시나리오 2 → 또 다른 가상의 252일
# ...
# 시나리오 10000 → 또 다른 가상의 252일


# ==================================================
# 현재 SK하이닉스 주가 가져오기
# ==================================================

initial_price = df["Close"].iloc[-1]

# df["Close"]
# → SK하이닉스 종가 데이터
#
# iloc[-1]
# → 가장 마지막 데이터
# → 즉 가장 최근 SK하이닉스 종가
#
# 이 가격을 모든 시뮬레이션의 출발가격으로 사용한다.

print()
print("현재 SK하이닉스 가격:", initial_price)


# ==================================================
# 난수 생성기 설정
# ==================================================

rng = np.random.default_rng(42)

# Monte Carlo는 랜덤하게 미래를 만든다.
#
# 그런데 실행할 때마다 완전히 다른 숫자가 나오면
# 공부하면서 결과를 비교하기 불편하다.
#
# 42라는 seed를 고정하면
# 코드를 다시 실행해도 같은 랜덤 결과를 재현할 수 있다.
#
# 나중에 seed를 없애면 실행할 때마다
# 조금씩 다른 Monte Carlo 결과가 나온다.


# ==================================================
# 핵심
# t-분포에서 가상의 일별수익률 생성
# ==================================================
daily_returns_t = t.rvs(

    df=df_fit,

    loc=loc_fit,

    scale=scale_fit,

    size=(n_sims, n_days),

    random_state=rng
)


# ==================================================
# 여기 정말 중요
# ==================================================

# 위에서 이미
#
# df_fit, loc_fit, scale_fit = t.fit(returns_1000)
#
# 을 실행했다.
#
# 즉 실제 SK하이닉스 일별수익률을 보고
#
# df_fit
# → 꼬리 두께
#
# loc_fit
# → 수익률 분포의 중심 위치
#
# scale_fit
# → 수익률 분포가 퍼져 있는 정도
#
# 를 찾아놓은 상태다.


# t.rvs()
# → Random Variates
# → t-분포에서 실제 랜덤 숫자를 뽑는 함수
#
# t.pdf()와 헷갈리지 말자.
#
# t.pdf()
# → 그래프를 그리기 위해 각 x에서 곡선의 "높이"를 계산
#
# t.rvs()
# → Monte Carlo에 사용할 실제 랜덤 수익률을 생성


# size=(n_sims, n_days)
#
# 현재:
#
# size=(10000, 252)
#
# 이므로
#
# 10000개의 미래 × 각각 252일
#
# 총 2,520,000개의 가상 일별수익률이 만들어진다.


print()
print("가상 일별수익률 배열 크기:")
print(daily_returns_t.shape)

# 결과:
# (10000, 252)


# ==================================================
# 배열 구조를 이해하면 중요하다
# ==================================================

# daily_returns_t는 대략 이런 구조다.
#
#                    1일      2일      3일   ...   252일
#
# 시나리오 1        +1.2%    -0.8%    +0.4%  ...   +0.2%
# 시나리오 2        -2.1%    +0.3%    -1.0%  ...   +1.1%
# 시나리오 3        +0.7%    +2.1%    -0.3%  ...   -0.5%
# ...
# 시나리오 10000     ...
#
#
# 한 "행(row)"이
# 하나의 가상 미래 1년이다.


# ==================================================
# 혹시 -100% 이하 수익률이 생성됐는지 확인
# ==================================================

invalid_returns = np.sum(daily_returns_t <= -1)

print()
print("-100% 이하로 생성된 일별수익률 개수:", invalid_returns)

# 단순수익률에서 -100%보다 작은 수익률은 현실적으로 불가능하다.
#
# 예:
# -100% → 주가가 0원이 됨
# -120% → 주가가 음수가 된다는 뜻 → 현실적으로 불가능
#
# t분포는 이론적으로 양쪽 끝이 무한히 열려 있기 때문에
# 아주 낮은 확률로 이런 값이 만들어질 수도 있다.
#
# 만약 이 숫자가 0이 아니라 자주 나온다면
# 단순수익률 t모형보다
# log-return 기반 모델로 바꾸는 것이 더 적절하다.


# ==================================================
# 각 시나리오의 주가 경로 만들기
# ==================================================

price_paths_t = initial_price * np.cumprod(

    1 + daily_returns_t,

    axis=1
)


# ==================================================
# 이 코드의 원리
# ==================================================

# daily_returns_t가 예를 들어
#
# 1일차 = +2%
# 2일차 = -1%
# 3일차 = +3%
#
# 라고 해보자.


# 1 + daily_returns_t
#
# 를 하면
#
# +2% → 1.02
# -1% → 0.99
# +3% → 1.03
#
# 로 바뀐다.


# np.cumprod()
# → cumulative product
# → 앞에서부터 계속 누적해서 곱한다.
#
# 따라서
#
# 1일차:
# 1.02
#
# 2일차:
# 1.02 × 0.99
#
# 3일차:
# 1.02 × 0.99 × 1.03
#
# 이런 식으로 주가가 복리로 움직인다.


# initial_price *
#
# 를 앞에 붙이면
#
# 현재 실제 SK하이닉스 가격에서 시작해서
# 미래 가격이 만들어진다.


# axis=1
#
# → 각 행(row)을 따라서 계산하라는 뜻
#
# 즉
#
# 시나리오 1의 252일을 누적
# 시나리오 2의 252일을 누적
# ...
# 시나리오 10000의 252일을 누적
#
# 하는 것이다.


# ==================================================
# 시작 가격까지 그래프에 포함시키기
# ==================================================

price_paths_t_with_start = np.column_stack(

    (
        np.full(n_sims, initial_price),
        price_paths_t
    )
)

# 원래 price_paths_t는
# "1일 뒤 가격"부터 시작한다.
#
# 그래서 그래프의 0일차에
# 현재가격(initial_price)을 하나 추가한 것이다.
#
# 이제:
#
# 0일 → 현재 가격
# 1일 → 첫 번째 가상 가격
# ...
# 252일 → 1년 뒤 가상 가격


# ==================================================
# Monte Carlo 주가 경로 시각화
# ==================================================

plt.figure(figsize=(12, 7))

# 시뮬레이션은 10000개를 했지만
# 10000개를 전부 그래프에 그리면 너무 복잡하다.
#
# 그래서 그래프에는 앞의 100개만 보여준다.
#
# 계산 자체는 여전히 10000개를 사용한다.

plt.plot(

    price_paths_t_with_start[:100].T,

    alpha=0.3,

    color="green"
)


plt.title(
    "SK Hynix t-Distribution Monte Carlo Simulation"
)

plt.xlabel("Trading Days")

plt.ylabel("Price")

plt.grid(alpha=0.3)

plt.show()


# ==================================================
# 1년 뒤 최종 가격 10000개 가져오기
# ==================================================

final_t = price_paths_t[:, -1]

# price_paths_t[:, -1]
#
# :
# → 모든 시나리오를 가져오고
#
# -1
# → 각 시나리오의 마지막 날을 가져온다.
#
#
# 따라서 final_t에는
#
# 시나리오 1의 1년 뒤 가격
# 시나리오 2의 1년 뒤 가격
# ...
# 시나리오 10000의 1년 뒤 가격
#
# 총 10000개가 들어있다.


print()
print("=== t분포 Monte Carlo : 1년 뒤 예상 가격 ===")

print(
    f"평균 가격: {final_t.mean():,.0f}원"
)

print(
    f"최저 가격: {final_t.min():,.0f}원"
)

print(
    f"최고 가격: {final_t.max():,.0f}원"
)

print(
    f"중앙값 가격: {np.median(final_t):,.0f}원"
)


# ==================================================
# 1년 뒤 가격의 백분위수 계산
# ==================================================

p5 = np.percentile(final_t, 5)

p25 = np.percentile(final_t, 25)

p50 = np.percentile(final_t, 50)

p75 = np.percentile(final_t, 75)

p95 = np.percentile(final_t, 95)


# percentile = 백분위수
#
# p5
# → 10000개의 결과를 낮은 가격부터 순서대로 세웠을 때
# → 하위 5% 지점
#
# p50
# → 정확히 가운데
# → 중앙값(median)
#
# p95
# → 상위 5%가 시작되는 지점


print()
print("=== 1년 뒤 가격 분포 ===")

print(
    f"시뮬레이션 결과의 가운데 90% 범위: "
    f"{p5:,.0f}원 ~ {p95:,.0f}원"
)

print(
    f"시뮬레이션 결과의 가운데 50% 범위: "
    f"{p25:,.0f}원 ~ {p75:,.0f}원"
)

print(
    f"중앙값: {p50:,.0f}원"
)


# ==================================================
# 1년 뒤 수익률 계산
# ==================================================

final_returns_t = (

    final_t / initial_price

) - 1


# 예를 들어
#
# 현재가격 = 200,000원
# 1년뒤가격 = 220,000원
#
# 이라면
#
# 220000 / 200000 - 1
#
# = 0.10
#
# → +10% 수익률


# final_returns_t 안에는
#
# 시나리오 1의 1년 수익률
# 시나리오 2의 1년 수익률
# ...
# 시나리오 10000의 1년 수익률
#
# 이 들어간다.


# ==================================================
# VaR(95%) 계산
# ==================================================

return_5_percentile = np.percentile(

    final_returns_t,

    5
)


# 수익률 10000개를
# 가장 나쁜 것부터 정렬한다고 생각하면 된다.
#
# 하위 5% 경계에 해당하는 수익률을 찾는다.


print()
print("=== 1년 VaR 분석 ===")

print(
    f"수익률 하위 5% 경계: "
    f"{return_5_percentile:.2%}"
)


# 예를 들어
#
# 수익률 하위 5% 경계 = -25%
#
# 라고 나오면
#
# 우리가 사용한 t분포 모델에서
# 1년 미래를 10000번 만들어봤을 때
#
# 약 5%의 시나리오는
# -25%보다 더 나쁜 결과가 나왔다는 뜻이다.


# 손실 크기로 VaR를 표현하고 싶다면

var_95 = -return_5_percentile

print(
    f"VaR(95%) 손실 크기: "
    f"{var_95:.2%}"
)


# ==================================================
# 최종 가격 분포 히스토그램
# ==================================================

plt.figure(figsize=(10, 6))

plt.hist(

    final_t,

    bins=50,

    density=True,

    alpha=0.7
)


plt.axvline(

    p5,

    linestyle="--",

    label="5th Percentile"
)


plt.axvline(

    p50,

    linestyle="--",

    label="Median"
)


plt.axvline(

    p95,

    linestyle="--",

    label="95th Percentile"
)


plt.title(
    "SK Hynix 1-Year Final Price Distribution"
)

plt.xlabel("Final Price")

plt.ylabel("Density")

plt.legend()

plt.grid(alpha=0.3)

plt.show()