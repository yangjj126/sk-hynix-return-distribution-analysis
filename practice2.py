# ==================================================
# SK하이닉스 리스크 분석 - 최종판
# 데이터 수집 → 로그수익률 → t분포 fitting
# → 몬테카를로 (시나리오 3종 × 기간 3종) → VaR/cVaR
# ==================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import t, norm

# ── 1. 데이터 수집 ──
ticker = yf.Ticker("000660.KS")
df = ticker.history(period="10y", auto_adjust=True)

print("===== 데이터 개요 =====")
print(f"데이터 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
print(f"전체 데이터 개수: {len(df)}")

# ── 2. 로그수익률 계산 ──
simple_returns = df["Close"].pct_change(fill_method=None).dropna()
log_returns = np.log(1 + simple_returns)

mu_log = log_returns.mean()
sigma_log = log_returns.std()

print()
print("===== 기본 통계 (로그수익률) =====")
print(f"평균 일일 로그수익률: {mu_log:.5f} ({mu_log*100:.3f}%)")
print(f"일일 표준편차: {sigma_log:.5f} ({sigma_log*100:.3f}%)")

# ── 3. t분포 fitting ──
df_fit, loc_fit, scale_fit = t.fit(log_returns)

print()
print("===== t분포 fitting =====")
print(f"추정된 자유도(df): {df_fit:.2f}")
print(f"추정된 중심(loc): {loc_fit:.5f}")
print(f"추정된 척도(scale): {scale_fit:.5f}")
if df_fit < 5:
    print("→ df < 5 : 심한 fat tail. 정규분포 가정은 리스크를 과소평가함")





# 이제 t분포내부로 드가자.



# ── 4. 몬테카를로 설정 ──
n_sims = 10000
# 시나리오 갯수, - t분포 내부에서의 시나리오임..
# 예를 들자면, t분포에 나온 그 수익률의 확률에 따라, 시나리오 각각에 일별 수익률이 배치되는것
initial_price = df["Close"].iloc[-1]
# Close컬럼의 마지막 값, 오늘 종가 - 모든 시뮬레이션의 출발점
horizons = {"1개월(21일)": 21, "3개월(63일)": 63, "1년(252일)": 252}
# horizons : 기간, 

risk_free_daily = 0.03 / 252   # 연 3% 무위험이자율 가정
# 무위험 이자율(보통 미국채금리) - 연3%를 1년치기간으로 나누어서, 무위험이자율 구한다

mu_scenarios = {
    "과거평균": mu_log,
    "드리프트제로": 0.0,
    "무위험이자율": risk_free_daily,
}
# mu_scenarios: mu문제를 해결하기 위해서 만든 3가지 시나리오
# 과거 평균하나만 보지말고, 여러개의 평균을 나란히 보자
# - 과거평균, 드리프트제로, 무위험이자율 - 이 세가지로 살펴보겠다는 뜻



# ── 5. 시뮬레이션 함수 ──
def run_simulation(mu_used, n_days):
    """t분포 기반 로그수익률 몬테카를로 → 최종가격 배열 반환"""
    Z_raw = t.rvs(df=df_fit, size=(n_sims, n_days))
    Z = Z_raw / np.sqrt(df_fit / (df_fit - 2))              # 분산 1로 정규화
    log_r = mu_used + scale_fit * np.sqrt(df_fit / (df_fit - 2)) * Z
    final_prices = initial_price * np.exp(log_r.sum(axis=1))
    return final_prices


# ── 6. 전체 조합 실행 ──
print()
print(f"현재가: {initial_price:,.0f}원")
print(f"시뮬레이션 횟수: {n_sims:,}회")
print("=" * 70)

for h_name, n_days in horizons.items():
    print(f"\n███ {h_name} ███")
    for s_name, mu_used in mu_scenarios.items():
        final = run_simulation(mu_used, n_days)
        pct = lambda q: (np.percentile(final, q) / initial_price - 1) * 100

        var95 = pct(5)
        cvar95 = (final[final <= np.percentile(final, 5)].mean()
                  / initial_price - 1) * 100

        print(f"  [{s_name}]")
        print(f"    50% 구간: {pct(25):+.1f}% ~ {pct(75):+.1f}%   "
              f"90% 구간: {pct(5):+.1f}% ~ {pct(95):+.1f}%")
        print(f"    VaR(95%): {var95:+.1f}%   cVaR(95%): {cvar95:+.1f}%")

# ── 7. 시각화: 드리프트제로 기준 1년 가격 범위 부채꼴 ──
n_days_plot = 252
Z_raw = t.rvs(df=df_fit, size=(n_sims, n_days_plot))
Z = Z_raw / np.sqrt(df_fit / (df_fit - 2))
log_r = 0.0 + scale_fit * np.sqrt(df_fit / (df_fit - 2)) * Z
price_paths = initial_price * np.exp(np.cumsum(log_r, axis=1))

days_axis = np.arange(n_days_plot)
p5 = np.percentile(price_paths, 5, axis=0)
p25 = np.percentile(price_paths, 25, axis=0)
p50 = np.percentile(price_paths, 50, axis=0)
p75 = np.percentile(price_paths, 75, axis=0)
p95 = np.percentile(price_paths, 95, axis=0)

plt.figure(figsize=(12, 6))
plt.fill_between(days_axis, p5, p95, alpha=0.2, color="green", label="90% range")
plt.fill_between(days_axis, p25, p75, alpha=0.35, color="green", label="50% range")
plt.plot(days_axis, p50, color="darkgreen", linewidth=2, label="Median")
plt.axhline(initial_price, color="gray", linestyle="--", label="Current price")

plt.title("SK Hynix 1-Year Price Range (t-dist, zero drift, log return)")
plt.xlabel("Days")
plt.ylabel("Price (KRW)")
plt.legend()
plt.tight_layout()
plt.show()

