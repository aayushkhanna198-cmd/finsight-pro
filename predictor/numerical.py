"""
numerical.py — Core Numerical Methods Engine
FinSight Pro | Financial Forecasting Platform

Implements:
  - Bisection Method (root-finding)
  - Newton-Raphson Method (root-finding)
  - Linear Regression (revenue prediction)
  - NPV — Net Present Value
  - IRR — Internal Rate of Return
  - Monte Carlo Simulation
  - Exponential Smoothing
"""

import numpy as np
import math


# ─────────────────────────────────────────────
# Profit / Revenue Models
# ─────────────────────────────────────────────

def profit(x, a=2, b=100, c=1000):
    """
    Quadratic profit function: P(x) = ax² - bx + c
    Default: P(x) = 2x² - 100x + 1000
    Break-even: P(x) = 0
    """
    return a * x**2 - b * x + c


def profit_derivative(x, a=2, b=100):
    """
    Derivative of profit function: P'(x) = 2ax - b
    Used by Newton-Raphson for tangent line calculation.
    """
    return 2 * a * x - b


# ─────────────────────────────────────────────
# Root-Finding: Bisection Method
# ─────────────────────────────────────────────

def bisection(f, a, b, tol=0.0001, max_iter=100):
    """
    Bisection Method — Root Finding

    Algorithm:
      1. Check f(a) * f(b) < 0 (root exists in [a, b])
      2. Compute midpoint c = (a + b) / 2
      3. If f(c) == 0 or interval < tol: return c
      4. Replace endpoint with same sign as f(c), repeat

    Convergence: O(log((b-a)/tol)) iterations
    Guaranteed to converge if f is continuous and f(a)*f(b) < 0

    Returns: dict with root, iterations, convergence_data
    """
    if f(a) * f(b) > 0:
        raise ValueError(f"No sign change in [{a}, {b}]. Bisection requires f(a)*f(b) < 0.")

    iterations = []
    for i in range(max_iter):
        c = (a + b) / 2.0
        fc = f(c)
        error = abs(b - a) / 2.0

        iterations.append({
            "n": i + 1,
            "a": round(a, 6),
            "b": round(b, 6),
            "c": round(c, 6),
            "f_c": round(fc, 6),
            "error": round(error, 8)
        })

        if fc == 0 or error < tol:
            return {
                "root": round(c, 6),
                "iterations": i + 1,
                "converged": True,
                "table": iterations,
                "method": "Bisection"
            }

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    return {
        "root": round((a + b) / 2, 6),
        "iterations": max_iter,
        "converged": False,
        "table": iterations,
        "method": "Bisection"
    }


# ─────────────────────────────────────────────
# Root-Finding: Newton-Raphson Method
# ─────────────────────────────────────────────

def newton_raphson(f, df, x0, tol=0.0001, max_iter=100):
    """
    Newton-Raphson Method — Root Finding

    Formula: x_{n+1} = x_n - f(x_n) / f'(x_n)

    Algorithm:
      1. Start at initial guess x0
      2. Compute tangent line at (x0, f(x0))
      3. Find x-intercept of tangent: x1 = x0 - f(x0)/f'(x0)
      4. Repeat until |x_{n+1} - x_n| < tolerance

    Convergence: Quadratic (much faster than Bisection)
    Requires: f'(x) ≠ 0 at iterates

    Returns: dict with root, iterations, convergence_data
    """
    x = float(x0)
    iterations = []

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-12:
            raise ValueError(f"Derivative near zero at x={x}. Newton-Raphson failed.")

        x_new = x - fx / dfx
        error = abs(x_new - x)

        iterations.append({
            "n": i + 1,
            "x_n": round(x, 6),
            "f_xn": round(fx, 6),
            "df_xn": round(dfx, 6),
            "x_new": round(x_new, 6),
            "error": round(error, 8)
        })

        if error < tol:
            return {
                "root": round(x_new, 6),
                "iterations": i + 1,
                "converged": True,
                "table": iterations,
                "method": "Newton-Raphson"
            }
        x = x_new

    return {
        "root": round(x, 6),
        "iterations": max_iter,
        "converged": False,
        "table": iterations,
        "method": "Newton-Raphson"
    }


# ─────────────────────────────────────────────
# Regression — Revenue Prediction
# ─────────────────────────────────────────────

def linear_regression(x_data, y_data):
    """
    Linear Regression — Least Squares Method

    Model: y = mx + b
    Normal equations:
      m = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
      b = (Σy - m*Σx) / n

    Returns coefficients, R², predictions for next 3 periods
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    n = len(x)

    # Compute using numpy polyfit (degree 1)
    coeffs = np.polyfit(x, y, 1)
    m, b = coeffs[0], coeffs[1]

    # Predictions
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

    # Forecast next 3 periods
    future_x = [max(x) + 1, max(x) + 2, max(x) + 3]
    future_y = [round(float(np.polyval(coeffs, fx)), 2) for fx in future_x]

    return {
        "slope": round(float(m), 4),
        "intercept": round(float(b), 4),
        "r_squared": round(float(r_squared), 4),
        "predictions": list(zip(future_x, future_y)),
        "fitted_values": [round(float(v), 2) for v in y_pred],
        "equation": f"y = {round(float(m),2)}x + {round(float(b),2)}"
    }


def polynomial_regression(x_data, y_data, degree=2):
    """
    Polynomial Regression

    Model: y = a_n*x^n + ... + a_1*x + a_0
    Fits using numpy polyfit with specified degree.
    """
    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)

    coeffs = np.polyfit(x, y, degree)
    y_pred = np.polyval(coeffs, x)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

    future_x = [max(x) + 1, max(x) + 2, max(x) + 3]
    future_y = [round(float(np.polyval(coeffs, fx)), 2) for fx in future_x]

    return {
        "degree": degree,
        "coefficients": [round(float(c), 4) for c in coeffs],
        "r_squared": round(float(r_squared), 4),
        "predictions": list(zip(future_x, future_y)),
        "fitted_values": [round(float(v), 2) for v in y_pred]
    }


# ─────────────────────────────────────────────
# Time Value of Money
# ─────────────────────────────────────────────

def npv(rate, cashflows):
    """
    Net Present Value

    Formula: NPV = Σ [C_t / (1 + r)^t]   for t = 0 to n

    Where:
      C_t = cash flow at time t
      r   = discount rate (as decimal, e.g. 0.10 for 10%)
      t   = time period

    A positive NPV means the investment adds value.
    """
    if rate <= -1:
        raise ValueError("Discount rate must be > -1")

    total = 0.0
    breakdown = []
    for t, cf in enumerate(cashflows):
        discount_factor = (1 + rate) ** t
        pv = cf / discount_factor
        total += pv
        breakdown.append({
            "period": t,
            "cashflow": cf,
            "discount_factor": round(discount_factor, 6),
            "present_value": round(pv, 4)
        })

    return {
        "npv": round(total, 4),
        "rate": rate,
        "breakdown": breakdown,
        "decision": "Accept (NPV > 0)" if total > 0 else "Reject (NPV ≤ 0)"
    }


def irr(cashflows, tol=0.0001, max_iter=1000):
    """
    Internal Rate of Return — Newton-Raphson on NPV function

    IRR is the rate r such that NPV(r) = 0

    Algorithm: Uses Newton-Raphson with:
      f(r)  = NPV(r)
      f'(r) = -Σ [t * C_t / (1+r)^(t+1)]

    Returns: IRR as decimal (multiply by 100 for percentage)
    """
    def npv_func(r):
        return sum(cf / (1 + r) ** t for t, cf in enumerate(cashflows))

    def npv_deriv(r):
        return sum(-t * cf / (1 + r) ** (t + 1) for t, cf in enumerate(cashflows))

    r = 0.1  # Initial guess: 10%
    for _ in range(max_iter):
        f_r = npv_func(r)
        df_r = npv_deriv(r)
        if abs(df_r) < 1e-12:
            break
        r_new = r - f_r / df_r
        if abs(r_new - r) < tol:
            return round(r_new * 100, 4)
        r = r_new

    return round(r * 100, 4)


# ─────────────────────────────────────────────
# Monte Carlo Simulation
# ─────────────────────────────────────────────

def monte_carlo_revenue(base_revenue, growth_mean, growth_std, periods=5, simulations=10000):
    """
    Monte Carlo Revenue Simulation

    Models stochastic revenue growth using normal distribution:
      R_t = R_{t-1} * (1 + N(μ, σ))

    Where:
      μ = mean growth rate
      σ = standard deviation of growth rate

    Returns: percentile bands (5th, 25th, 50th, 75th, 95th)
    """
    np.random.seed(42)
    all_paths = []

    for _ in range(simulations):
        path = [base_revenue]
        for _ in range(periods):
            growth = np.random.normal(growth_mean, growth_std)
            path.append(path[-1] * (1 + growth))
        all_paths.append(path)

    all_paths = np.array(all_paths)
    result = {}
    for p in [5, 25, 50, 75, 95]:
        result[f"p{p}"] = [round(float(v), 2) for v in np.percentile(all_paths, p, axis=0)]

    return {
        "percentiles": result,
        "periods": list(range(periods + 1)),
        "expected_final": round(float(np.mean(all_paths[:, -1])), 2),
        "std_final": round(float(np.std(all_paths[:, -1])), 2),
        "simulations": simulations
    }


# ─────────────────────────────────────────────
# Exponential Smoothing
# ─────────────────────────────────────────────

def exponential_smoothing(data, alpha=0.3, periods_ahead=3):
    """
    Simple Exponential Smoothing

    Formula: S_t = α * Y_t + (1 - α) * S_{t-1}

    Where:
      α     = smoothing factor (0 < α < 1)
      Y_t   = actual value at time t
      S_t   = smoothed value at time t

    High α → more weight on recent observations
    Low  α → more weight on historical data
    """
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be between 0 and 1")

    smoothed = [data[0]]
    for i in range(1, len(data)):
        s = alpha * data[i] + (1 - alpha) * smoothed[-1]
        smoothed.append(round(s, 4))

    # Forecast (flat projection from last smoothed value)
    last_smooth = smoothed[-1]
    forecast = [round(last_smooth, 4)] * periods_ahead

    return {
        "alpha": alpha,
        "smoothed": smoothed,
        "forecast": forecast,
        "mae": round(float(np.mean(np.abs(np.array(data) - np.array(smoothed)))), 4)
    }


# ─────────────────────────────────────────────
# Payback Period
# ─────────────────────────────────────────────

def payback_period(initial_investment, annual_cashflows):
    """
    Payback Period Calculation

    Simple method: counts years until cumulative cash flow ≥ investment.
    Fractional year = remaining balance / next year CF

    Returns: years to recover investment
    """
    cumulative = 0.0
    for i, cf in enumerate(annual_cashflows):
        cumulative += cf
        if cumulative >= initial_investment:
            # Fractional year
            prev_cumulative = cumulative - cf
            fraction = (initial_investment - prev_cumulative) / cf
            return round(i + fraction, 2)
    return None  # Never recovered
