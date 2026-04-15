"""
views.py — Application Views
FinSight Pro | Financial Forecasting Platform

Routes:
  /                   → landing page (public)
  /dashboard/         → main dashboard (login required)
  /analysis/          → run analysis (login required)
  /results/           → view results (login required)
  /history/           → analysis history (login required)
  /register/          → user registration
  /login/             → user login
  /logout/            → logout
  /api/analyze/       → JSON API endpoint
"""

import json
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

from .forms import RegisterForm, AnalysisForm
from .models import AnalysisHistory, UserProfile
from .numerical import (
    profit, profit_derivative,
    bisection, newton_raphson,
    linear_regression, polynomial_regression,
    npv, irr,
    monte_carlo_revenue,
    exponential_smoothing,
    payback_period,
)

# ─────────────────────────────────────
# Chart Helpers
# ─────────────────────────────────────

CHART_STYLE = {
    "bg": "#0d1117",
    "fg": "#e6edf3",
    "grid": "#21262d",
    "accent1": "#58a6ff",
    "accent2": "#3fb950",
    "accent3": "#d2a8ff",
    "accent4": "#ffa657",
    "danger": "#f85149",
}

def _fig_to_b64(fig):
    """Convert matplotlib figure to base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=CHART_STYLE["bg"], dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64

def _setup_ax(ax):
    """Apply dark theme to matplotlib axes."""
    ax.set_facecolor(CHART_STYLE["bg"])
    ax.tick_params(colors=CHART_STYLE["fg"], labelsize=9)
    ax.xaxis.label.set_color(CHART_STYLE["fg"])
    ax.yaxis.label.set_color(CHART_STYLE["fg"])
    for spine in ax.spines.values():
        spine.set_edgecolor(CHART_STYLE["grid"])
    ax.grid(True, color=CHART_STYLE["grid"], linewidth=0.6, linestyle="--", alpha=0.7)
    ax.set_title(ax.get_title(), color=CHART_STYLE["fg"], fontsize=10, pad=10)


def _chart_breakeven(x_data, a=2, b=100, c=1000):
    """Generate break-even profit curve chart."""
    x = np.linspace(min(x_data) - 5, max(x_data) + 5, 400)
    y = a * x**2 - b * x + c

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=CHART_STYLE["bg"])
    ax.plot(x, y, color=CHART_STYLE["accent1"], linewidth=2, label="Profit P(x)")
    ax.axhline(0, color=CHART_STYLE["fg"], linewidth=0.8, linestyle="-", alpha=0.5)

    # Mark break-even points
    for bx in x_data:
        ax.axvline(bx, color=CHART_STYLE["danger"], linewidth=1.2, linestyle="--", alpha=0.8)
        ax.plot(bx, 0, "o", color=CHART_STYLE["danger"], markersize=8, zorder=5)
        ax.annotate(f"x={bx:.2f}", xy=(bx, 0), xytext=(bx + 0.5, c * 0.15),
                    color=CHART_STYLE["danger"], fontsize=8, arrowprops=dict(arrowstyle="-", color=CHART_STYLE["danger"], lw=0.8))

    ax.fill_between(x, y, 0, where=(y >= 0), alpha=0.12, color=CHART_STYLE["accent2"], label="Profit zone")
    ax.fill_between(x, y, 0, where=(y < 0), alpha=0.12, color=CHART_STYLE["danger"], label="Loss zone")
    ax.set_xlabel("Units (x)")
    ax.set_ylabel("Profit P(x)")
    ax.set_title("Break-Even Analysis — Profit Function")
    ax.legend(facecolor=CHART_STYLE["bg"], edgecolor=CHART_STYLE["grid"], labelcolor=CHART_STYLE["fg"], fontsize=8)
    _setup_ax(ax)
    return _fig_to_b64(fig)


def _chart_regression(x_data, y_data, reg_result, future_x):
    """Revenue trend + regression line chart."""
    x = np.array(x_data)
    y = np.array(y_data)
    m = reg_result["slope"]
    b = reg_result["intercept"]

    x_ext = np.array(list(x) + list(future_x))
    y_line = m * x_ext + b
    future_y = [v for _, v in reg_result["predictions"]]

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=CHART_STYLE["bg"])
    ax.scatter(x, y, color=CHART_STYLE["accent1"], s=50, zorder=5, label="Actual")
    ax.plot(x, m * x + b, color=CHART_STYLE["accent3"], linewidth=1.8, linestyle="--", label=f"Trend: {reg_result['equation']}")
    ax.scatter(future_x, future_y, color=CHART_STYLE["accent2"], s=60, marker="^", zorder=5, label="Forecast")
    ax.plot(list(x)[-1:] + list(future_x), [m * x[-1] + b] + future_y,
            color=CHART_STYLE["accent2"], linewidth=1.5, linestyle=":", alpha=0.8)
    ax.set_xlabel("Period")
    ax.set_ylabel("Revenue")
    ax.set_title(f"Revenue Trend & Forecast  (R² = {reg_result['r_squared']})")
    ax.legend(facecolor=CHART_STYLE["bg"], edgecolor=CHART_STYLE["grid"], labelcolor=CHART_STYLE["fg"], fontsize=8)
    _setup_ax(ax)
    return _fig_to_b64(fig)


def _chart_npv_waterfall(breakdown):
    """NPV waterfall chart showing discounted cash flows."""
    periods = [b["period"] for b in breakdown]
    pvs = [b["present_value"] for b in breakdown]

    colors = [CHART_STYLE["danger"] if v < 0 else CHART_STYLE["accent2"] for v in pvs]

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=CHART_STYLE["bg"])
    bars = ax.bar(periods, pvs, color=colors, edgecolor=CHART_STYLE["bg"], linewidth=0.5, alpha=0.9)
    ax.axhline(0, color=CHART_STYLE["fg"], linewidth=0.7, alpha=0.5)

    for bar, val in zip(bars, pvs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(pvs) * 0.02),
                f"{val:,.0f}", ha="center", va="bottom", color=CHART_STYLE["fg"], fontsize=8)

    ax.set_xlabel("Period")
    ax.set_ylabel("Present Value")
    ax.set_title("Discounted Cash Flows (NPV Breakdown)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _setup_ax(ax)
    return _fig_to_b64(fig)


def _chart_monte_carlo(mc_result):
    """Monte Carlo fan chart."""
    p = mc_result["percentiles"]
    t = mc_result["periods"]

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor=CHART_STYLE["bg"])
    ax.fill_between(t, p["p5"], p["p95"], alpha=0.15, color=CHART_STYLE["accent1"], label="5–95th pct")
    ax.fill_between(t, p["p25"], p["p75"], alpha=0.25, color=CHART_STYLE["accent1"], label="25–75th pct")
    ax.plot(t, p["p50"], color=CHART_STYLE["accent1"], linewidth=2, label="Median")
    ax.set_xlabel("Period")
    ax.set_ylabel("Revenue")
    ax.set_title(f"Monte Carlo Simulation ({mc_result['simulations']:,} runs)")
    ax.legend(facecolor=CHART_STYLE["bg"], edgecolor=CHART_STYLE["grid"], labelcolor=CHART_STYLE["fg"], fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    _setup_ax(ax)
    return _fig_to_b64(fig)


# ─────────────────────────────────────
# Authentication Views
# ─────────────────────────────────────

def landing(request):
    """Public landing page."""
    if request.user.is_authenticated:
        return redirect("dashboard")
    return render(request, "predictor/landing.html")


def register_view(request):
    """User registration."""
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            UserProfile.objects.create(
                user=user,
                company=form.cleaned_data.get("company", ""),
                industry=form.cleaned_data.get("industry", ""),
            )
            login(request, user)
            messages.success(request, f"Welcome to FinSight, {user.first_name or user.username}!")
            return redirect("dashboard")
    else:
        form = RegisterForm()

    return render(request, "predictor/register.html", {"form": form})


def login_view(request):
    """User login."""
    if request.user.is_authenticated:
        return redirect("dashboard")

    if request.method == "POST":
        form = AuthenticationForm(data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            messages.success(request, f"Welcome back, {user.first_name or user.username}!")
            return redirect(request.GET.get("next", "dashboard"))
        else:
            messages.error(request, "Invalid credentials. Please try again.")
    else:
        form = AuthenticationForm()

    return render(request, "predictor/login.html", {"form": form})


def logout_view(request):
    """Logout."""
    logout(request)
    messages.info(request, "You have been signed out.")
    return redirect("landing")


# ─────────────────────────────────────
# Core Application Views
# ─────────────────────────────────────

@login_required
def dashboard(request):
    """Main dashboard — shows summary stats and recent history."""
    history = AnalysisHistory.objects.filter(user=request.user)[:5]
    total_analyses = AnalysisHistory.objects.filter(user=request.user).count()

    context = {
        "history": history,
        "total_analyses": total_analyses,
    }
    return render(request, "predictor/dashboard.html", context)


@login_required
def analysis_view(request):
    """Run a new analysis."""
    form = AnalysisForm()
    results = None
    charts = {}
    error = None

    if request.method == "POST":
        fixed_cost = float(request.POST.get('fixed_cost'))
        price = float(request.POST.get('price'))
        cost = float(request.POST.get('cost'))
        form = AnalysisForm(request.POST)
        if form.is_valid():
            try:
                data = {
                    'fixed_cost': fixed_cost,
                    'price': price,
                    'cost': cost
                }
                results, charts = _run_analysis(data)

                # Save to history
                AnalysisHistory.objects.create(
                    user=request.user,
                    analysis_type=form.cleaned_data["analysis_type"],
                    name=form.cleaned_data.get("analysis_name", "Untitled"),
                    input_data=form.cleaned_data,
                    result_data={k: v for k, v in results.items() if k != "charts"},
                )
                messages.success(request, "Analysis completed successfully.")

            except Exception as e:
                error = str(e)
                messages.error(request, f"Analysis error: {error}")

    return render(request, "predictor/analysis.html", {
        "form": form,
        "results": results,
        "charts": charts,
        "error": error,
    })


@login_required
def history_view(request):
    """View analysis history."""
    analyses = AnalysisHistory.objects.filter(user=request.user)
    return render(request, "predictor/history.html", {"analyses": analyses})


@login_required
def history_detail(request, pk):
    """View a specific past analysis."""
    analysis = get_object_or_404(AnalysisHistory, pk=pk, user=request.user)
    return render(request, "predictor/history_detail.html", {"analysis": analysis})


# ─────────────────────────────────────
# Analysis Engine
# ─────────────────────────────────────

def _run_analysis(data):
    """
    Core analysis dispatcher.
    Returns (results_dict, charts_dict)
    """
    analysis_type = data["analysis_type"]
    results = {"type": analysis_type}
    charts = {}

    # ── Break-Even Analysis ───────────────────
    if analysis_type in ("breakeven", "all"):
        f = lambda x: profit(x)
        df = lambda x: profit_derivative(x)

        bis = bisection(f, 0, 50)
        nwt = newton_raphson(f, df, x0=10)

        results["bisection"] = bis
        results["newton"] = nwt
        charts["breakeven"] = _chart_breakeven([bis["root"], nwt["root"]])

    # ── Revenue Forecast ─────────────────────
    if analysis_type in ("forecast", "all"):
        x_data = data.get("x_data", [1, 2, 3, 4, 5])
        y_data = data.get("y_data", [100, 150, 200, 260, 300])

        reg = linear_regression(x_data, y_data)
        future_x = [p for p, _ in reg["predictions"]]

        results["regression"] = reg
        charts["regression"] = _chart_regression(x_data, y_data, reg, future_x)

    # ── NPV Analysis ─────────────────────────
    if analysis_type in ("npv", "all"):
        rate = data.get("discount_rate", 0.10)
        cashflows = data.get("cashflows", [-1000, 300, 400, 500, 600])

        npv_result = npv(rate, cashflows)
        irr_result = irr(cashflows)
        pb = payback_period(abs(cashflows[0]), cashflows[1:])

        results["npv"] = npv_result
        results["irr"] = irr_result
        results["payback"] = pb
        charts["npv"] = _chart_npv_waterfall(npv_result["breakdown"])

    # ── Monte Carlo ───────────────────────────
    if analysis_type in ("monte_carlo", "all"):
        base = data.get("base_revenue", 1000000)
        mean = data.get("growth_mean", 0.08)
        std = data.get("growth_std", 0.05)

        mc = monte_carlo_revenue(base, mean, std)
        results["monte_carlo"] = mc
        charts["monte_carlo"] = _chart_monte_carlo(mc)

    return results, charts


# ─────────────────────────────────────
# JSON API
# ─────────────────────────────────────

@login_required
def analysis_view(request):
    """Run a new analysis."""
    form = AnalysisForm()
    results = None
    charts = {}
    error = None

    if request.method == "POST":
        form = AnalysisForm(request.POST)

        if form.is_valid():
            try:
                # DIRECT form data use karo (no manual float)
                results, charts = _run_analysis(form.cleaned_data)

                # Save to history
                AnalysisHistory.objects.create(
                    user=request.user,
                    analysis_type=form.cleaned_data["analysis_type"],
                    name=form.cleaned_data.get("analysis_name", "Untitled"),
                    input_data=form.cleaned_data,
                    result_data={k: v for k, v in results.items() if k != "charts"},
                )

                messages.success(request, "Analysis completed successfully.")

            except Exception as e:
                error = str(e)
                messages.error(request, f"Analysis error: {error}")

    return render(request, "predictor/analysis.html", {
        "form": form,
        "results": results,
        "charts": charts,
        "error": error,
    })
@login_required
def api_analyze(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        data = json.loads(request.body)
        results, _ = _run_analysis(data)
        return JsonResponse({"success": True, "results": results})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=400)