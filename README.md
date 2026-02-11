# 📊 Automated-General-Report-for-Portfolio-Risk-Analysis

This repository provides an advanced **Quantitative Risk Report** focused on evaluating the stability and efficiency of a diversified investment portfolio. The analysis utilizes high-frequency risk metrics and statistical tests to validate the portfolio's resilience against market shocks and systematic risk.

🎯 **Objective:** To implement a robust risk management framework using conditional volatility and tail risk metrics to optimize capital preservation.

---

## 📖 Extended Overview
This project contains a comprehensive quantitative evaluation of a multi-asset portfolio during a high-volatility period (**August 2025 to November 2025**). The analysis integrates performance attribution and advanced risk modeling—such as **Backtesting VaR** and **Rolling Kurtosis**—to identify regime changes in market risk.


### 🎯 Key Objectives of the Analysis
* **Dynamic Risk Measurement:** Monitoring of 21-day Rolling Volatility and Rolling Kurtosis to detect "Volatility Clustering" and leptokurtic shifts (tail risk).
* **Tail Risk & Expected Shortfall:** Calculation of **Value at Risk (VaR)** and **Conditional VaR (CVaR)** at 95% and 99% levels to quantify the severity of extreme loss scenarios.
* **Model Validation:** Implementation of VaR Backtesting with an **Empirical Failure Rate (4.35%)** to validate the statistical calibration of risk models.
* **Risk-Adjusted Efficiency:** Evaluation through **Sharpe (1.68)**, **Sortino (2.53)**, and **Calmar (4.69)** ratios to ensure superior return per unit of downside risk.
* **Performance Attribution:** Weekly asset contribution analysis and **Jensen's Alpha** calculation to isolate active management results.

---

## 🔍 Assets Analyzed
The portfolio employs a strategic mix of growth and defensive instruments:

* **🚀 Individual Equities (Alpha Drivers):** Apple (AAPL), Amazon (AMZN), JPMorgan (JPM), Goldman Sachs (GS).
* **🛡️ Fixed Income (Stabilizer):** Vanguard Total Bond Market ETF (BND), acting as a primary risk hedge with low/negative correlation to equities.
* **📊 Benchmarks:** SPDR S&P 500 (SPY) and Dow Jones Industrial Average (DIA).

---

## 📈 Key Portfolio Results
* **Cumulative Performance:** Total return of **6.66% (16.67% YTD)**, reaching an All-Time High (ATH) during the trend recovery phase.
* **Systematic Risk:** **Beta of 0.91** and **$R^2$ of 77.51%**, indicating lower sensitivity to market volatility than the SPY benchmark.
* **Alpha Generation:** **Jensen's Alpha of 7.36%**, confirming significant value creation above the benchmark.
* **Drawdown Management:** Maximum Drawdown contained at **-5.34%** with rapid recovery, demonstrating high liquidity and effective defensive positioning.

---

## 🛠️ Code Structure & Logic

### 1. Volatility & Regime Detection 🔍
* Uses rolling windows to identify regime shifts (e.g., transitions in annualized volatility).

### 2. Statistical Engine 🧬
* Performs **Jarque-Bera tests** and distribution analysis (Skewness: -0.27, Kurtosis: 0.47) to assess the viability of Gaussian vs. t-Student risk models.

### 3. Correlation & Diversification 🤝
* Generates an asset correlation matrix to monitor the **"BND buffer"** effect and sector-specific dependencies.

---

## 🚀 Technologies & Concepts Used
* **Quantitative Finance:** Modern Portfolio Theory (MPT), Risk Budgeting, and Performance Attribution.
* **Risk Modeling:** VaR/CVaR Rolling, Backtesting, and Volatility Clustering analysis.
* **Python Stack:** Pandas & NumPy (Matrix operations), Matplotlib & Seaborn (Heatmaps and Rolling charts).
* **Economic Context:** Integration of macro events (Fed cuts, trade tariffs) into quantitative price action analysis.

---

## ⚙️ Installation & Requirements

### 1. Clone the repository
```bash
git clone [https://github.com/tu-usuario/Automated-General-Report-for-Portfolio-Risk-Analysis.git](https://github.com/tu-usuario/Automated-General-Report-for-Portfolio-Risk-Analysis.git)
cd Automated-General-Report-for-Portfolio-Risk-Analysis
