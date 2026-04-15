# FinSight Pro — Financial Forecasting Platform
## Complete Project Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Numerical Methods — Deep Dive](#4-numerical-methods)
5. [Project Structure](#5-project-structure)
6. [Installation & Setup](#6-installation--setup)
7. [Database Models](#7-database-models)
8. [API Reference](#8-api-reference)
9. [Feature Walkthrough](#9-feature-walkthrough)
10. [Testing Guide](#10-testing-guide)
11. [Deployment Guide](#11-deployment-guide)
12. [Viva Q&A — Complete Reference](#12-viva-qa)

---

## 1. Project Overview

**FinSight Pro** is a full-stack web application that applies numerical methods to financial analytics. Built with Django (Python), it enables users to:

- Find **break-even points** using Bisection and Newton-Raphson root-finding methods
- **Forecast revenue** using linear and polynomial regression
- Calculate **NPV and IRR** for investment appraisal
- Run **Monte Carlo simulations** for stochastic risk modeling
- **Authenticate** with a secure signup/login system
- **Save and review** all past analyses via a personal history dashboard

The project demonstrates that rigorous mathematical techniques — traditionally confined to academic contexts — can be packaged into a usable, production-quality web product.

---

## 2. System Architecture

```
Browser (HTML/CSS/JS)
        │
        ▼ HTTP Request
┌─────────────────────────┐
│   Django Web Framework  │
│   ┌─────────────────┐   │
│   │  URL Router     │   │
│   │  predictor/urls │   │
│   └────────┬────────┘   │
│            │            │
│   ┌────────▼────────┐   │
│   │  Views Layer    │   │
│   │  views.py       │   │
│   └────────┬────────┘   │
│            │            │
│   ┌────────▼────────┐   │
│   │  Business Logic │   │
│   │  numerical.py   │   │  ◄── NumPy + SciPy
│   └────────┬────────┘   │
│            │            │
│   ┌────────▼────────┐   │
│   │  Data Layer     │   │
│   │  models.py      │   │  ◄── SQLite / PostgreSQL
│   └─────────────────┘   │
└─────────────────────────┘
        │
        ▼ HTTP Response
   Rendered HTML + base64 Charts
```

**Request Lifecycle:**
1. User submits form on `/analysis/`
2. Django CSRF middleware validates token
3. `AnalysisForm` cleans and validates input
4. `views._run_analysis()` dispatches to numerical methods
5. `numerical.py` computes results using NumPy
6. Matplotlib generates charts as base64 PNG
7. Results rendered into template, returned to browser
8. `AnalysisHistory` record saved to database

---

## 3. Technology Stack

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Backend framework | Django | 4.2+ | MVC web framework |
| Language | Python | 3.10+ | Core language |
| Numerical computing | NumPy | 1.24+ | Array math, polyfit |
| Visualization | Matplotlib | 3.7+ | Chart generation |
| Database | SQLite (dev) / PostgreSQL (prod) | — | Data persistence |
| Auth | Django built-in | — | User management |
| Frontend | HTML5 + CSS3 + Vanilla JS | — | UI layer |
| Fonts | Space Grotesk + JetBrains Mono | — | Typography |

---

## 4. Numerical Methods

### 4.1 Bisection Method

**Category:** Root-finding / Bracketing method

**Problem:** Find x such that f(x) = 0, i.e., the break-even units.

**Algorithm:**
```
INPUT: f, a, b, tol
REQUIRE: f(a) * f(b) < 0  (sign change → root in [a,b])

WHILE (b - a) >= tol:
    c = (a + b) / 2          ← midpoint
    IF f(c) == 0:
        RETURN c             ← exact root
    ELIF f(a) * f(c) < 0:
        b = c                ← root in left half
    ELSE:
        a = c                ← root in right half

RETURN (a + b) / 2
```

**Convergence:** Linear — O(log₂((b-a)/tol)) iterations
- Iteration n error: |eₙ| ≤ (b-a) / 2ⁿ
- For tol=0.001 and [0,50]: ceil(log₂(50/0.001)) = 16 iterations

**Pros:** Simple, robust, guaranteed convergence
**Cons:** Slower than Newton-Raphson; requires bracket

**Application in FinSight:**
```python
# Profit(x) = 2x² - 100x + 1000
# Find x where Profit = 0 (break-even)
result = bisection(profit, a=0, b=50, tol=0.0001)
# → root ≈ 25.0 (units where profit = 0)
```

---

### 4.2 Newton-Raphson Method

**Category:** Root-finding / Open method

**Formula:** `x_{n+1} = x_n - f(x_n) / f'(x_n)`

**Geometric intuition:**
- At point xₙ, draw the tangent to f(x)
- The tangent hits the x-axis at xₙ₊₁
- This new point is much closer to the root

**Algorithm:**
```
INPUT: f, f', x₀, tol
x = x₀

WHILE True:
    x_new = x - f(x) / f'(x)
    IF |x_new - x| < tol:
        RETURN x_new
    x = x_new
```

**Convergence:** Quadratic — errors roughly square each iteration
- If |e₀| = 0.1, then |e₁| ≈ 0.01, |e₂| ≈ 0.0001

**For Profit(x) = 2x² - 100x + 1000:**
- f(x)  = 2x² - 100x + 1000
- f'(x) = 4x - 100

**Iteration trace (starting x₀=10):**
| n | xₙ | f(xₙ) | f'(xₙ) | xₙ₊₁ | Error |
|---|-----|--------|---------|-------|-------|
| 1 | 10 | 700 | -60 | 21.667 | 11.667 |
| 2 | 21.667 | 38.89 | -13.33 | 24.583 | 2.917 |
| 3 | 24.583 | 1.736 | -1.667 | 24.998 | 0.415 |
| 4 | 24.998 | 0.001 | -0.008 | 25.000 | 0.002 |

**Pros:** Fast convergence, few iterations needed
**Cons:** Requires derivative; can fail if f'(x) ≈ 0

---

### 4.3 Linear Regression

**Category:** Statistical estimation / Supervised learning

**Model:** y = mx + b (least squares fit)

**Normal equations** (derived by minimizing Σ(yᵢ - ŷᵢ)²):
```
m = (n·Σxᵢyᵢ  −  Σxᵢ·Σyᵢ) / (n·Σxᵢ²  −  (Σxᵢ)²)
b = (Σyᵢ − m·Σxᵢ) / n
```

**R² (Coefficient of Determination):**
```
R² = 1 − (SS_res / SS_tot)
   where SS_res = Σ(yᵢ − ŷᵢ)²
         SS_tot = Σ(yᵢ − ȳ)²
```
- R² = 1.0 → perfect fit
- R² = 0.0 → model no better than mean

**Example (default data):**
```
x: [1, 2, 3, 4, 5]
y: [100, 150, 200, 260, 300]

→ m = 50.0, b = 46.0
→ Equation: y = 50.0x + 46.0
→ R² = 0.9966 (excellent fit)
→ Period 6 forecast: 346.0
→ Period 7 forecast: 396.0
→ Period 8 forecast: 446.0
```

**Implementation uses `numpy.polyfit`:**
```python
coeffs = np.polyfit(x, y, deg=1)   # [slope, intercept]
y_pred = np.polyval(coeffs, x_new) # evaluate at new x
```

---

### 4.4 Net Present Value (NPV)

**Category:** Time value of money / Investment appraisal

**Formula:**
```
NPV = Σ [Cₜ / (1 + r)ᵗ]    for t = 0, 1, 2, ..., n
```

Where:
- Cₜ = cash flow at period t (negative = outflow)
- r  = discount rate (e.g. 0.10 for 10%)
- t  = time period

**Decision rule:**
- NPV > 0 → Accept investment (adds value)
- NPV < 0 → Reject investment (destroys value)
- NPV = 0 → Indifferent (exactly meets hurdle rate)

**Example:**
```
Cashflows: [-1000, 300, 400, 500, 600]
Rate: 10% (r = 0.10)

t=0: -1000 / (1.10)⁰ = -1000.00
t=1:   300 / (1.10)¹ =   272.73
t=2:   400 / (1.10)² =   330.58
t=3:   500 / (1.10)³ =   375.66
t=4:   600 / (1.10)⁴ =   409.81

NPV = -1000 + 272.73 + 330.58 + 375.66 + 409.81
NPV = +388.78  →  ACCEPT
```

---

### 4.5 Internal Rate of Return (IRR)

**Category:** Root-finding applied to NPV

**Definition:** IRR is the rate r* such that NPV(r*) = 0

**Method:** Newton-Raphson applied to the NPV function:
```
f(r)  = NPV(r)  = Σ [Cₜ / (1+r)ᵗ]
f'(r) = dNPV/dr = Σ [-t·Cₜ / (1+r)^(t+1)]

r_{n+1} = rₙ - f(rₙ) / f'(rₙ)
```

**Starting guess:** r₀ = 0.10 (10%)
**Convergence:** Typically 5–15 iterations

---

### 4.6 Monte Carlo Simulation

**Category:** Stochastic simulation / Risk modeling

**Model:**
```
Rₜ = Rₜ₋₁ × (1 + εₜ)    where εₜ ~ N(μ, σ)
```

Where:
- Rₜ  = revenue at period t
- μ   = mean growth rate (e.g. 8%)
- σ   = standard deviation of growth (e.g. 5%)
- εₜ  = random draw from normal distribution

**Algorithm:**
```
FOR simulation in 1..10000:
    path = [base_revenue]
    FOR t in 1..5:
        growth = Normal(μ, σ)        ← random draw
        Rₜ = R_{t-1} * (1 + growth)
        path.append(Rₜ)
    store path

COMPUTE percentiles: 5th, 25th, 50th, 75th, 95th
```

**Output:** Fan chart showing uncertainty bands
- Dark band (25–75th): likely range
- Light band (5–95th): extreme range
- Center line: median expectation

---

## 5. Project Structure

```
fintech_project/
│
├── manage.py                    ← Django CLI entry point
├── db.sqlite3                   ← Development database
├── requirements.txt             ← Python dependencies
├── README.md                    ← This file
│
├── fintech_project/             ← Django project package
│   ├── __init__.py
│   ├── settings.py              ← Configuration (DB, apps, etc.)
│   ├── urls.py                  ← Root URL dispatcher
│   └── wsgi.py                  ← WSGI server entry point
│
├── predictor/                   ← Main Django app
│   ├── __init__.py
│   ├── models.py                ← AnalysisHistory, UserProfile
│   ├── views.py                 ← All view functions + chart builders
│   ├── forms.py                 ← RegisterForm, AnalysisForm
│   ├── urls.py                  ← App URL patterns
│   ├── numerical.py             ← All numerical method implementations
│   ├── admin.py                 ← Django admin registration
│   │
│   ├── templates/predictor/
│   │   ├── landing.html         ← Public homepage
│   │   ├── register.html        ← Sign-up page
│   │   ├── login.html           ← Sign-in page
│   │   ├── dashboard.html       ← Post-login dashboard
│   │   ├── analysis.html        ← Run analysis + results
│   │   ├── history.html         ← Analysis history list
│   │   └── history_detail.html  ← Single past analysis
│   │
│   └── static/predictor/
│       ├── css/                 ← (optional) extra CSS
│       └── js/                  ← (optional) extra JS
│
├── templates/
│   └── base.html                ← Global base template (nav, layout, styles)
│
└── static/
    └── css/                     ← Project-level static files
```

---

## 6. Installation & Setup

### Prerequisites

```bash
Python 3.10+
pip
virtualenv (recommended)
```

### Step-by-Step Setup

```bash
# 1. Clone / download the project
cd fintech_project/

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Apply database migrations
python manage.py makemigrations
python manage.py migrate

# 5. Create admin superuser (optional)
python manage.py createsuperuser

# 6. Run development server
python manage.py runserver

# 7. Visit in browser
http://127.0.0.1:8000/
```

### requirements.txt

```
Django>=4.2
numpy>=1.24
matplotlib>=3.7
```

### Environment Variables (Production)

```bash
# Create .env file
SECRET_KEY=your-random-50-char-string
DEBUG=False
DATABASE_URL=postgresql://user:pass@host:5432/dbname
ALLOWED_HOSTS=yourdomain.com
```

---

## 7. Database Models

### AnalysisHistory

Stores every analysis run by authenticated users.

| Field | Type | Description |
|-------|------|-------------|
| id | AutoField PK | Auto-generated |
| user | FK → User | Owner |
| analysis_type | CharField | breakeven / forecast / npv / etc. |
| name | CharField | Human-readable label |
| input_data | JSONField | All form inputs |
| result_data | JSONField | Computed results |
| created_at | DateTimeField | Timestamp |

### UserProfile

Extended user info beyond Django's built-in User model.

| Field | Type | Description |
|-------|------|-------------|
| id | AutoField PK | Auto-generated |
| user | OneToOne → User | Linked account |
| company | CharField | Optional company name |
| industry | CharField | Optional industry sector |
| created_at | DateTimeField | Profile creation time |

### Migrations

```bash
python manage.py makemigrations predictor
python manage.py migrate
```

---

## 8. API Reference

### Web Endpoints

| Method | URL | Auth | Description |
|--------|-----|------|-------------|
| GET | `/` | No | Landing page |
| GET/POST | `/register/` | No | User sign-up |
| GET/POST | `/login/` | No | User sign-in |
| GET | `/logout/` | No | Sign-out |
| GET | `/dashboard/` | ✓ | User dashboard |
| GET/POST | `/analysis/` | ✓ | Run analysis |
| GET | `/history/` | ✓ | Analysis history |
| GET | `/history/<id>/` | ✓ | Analysis detail |

### JSON API Endpoint

```
POST /api/analyze/
Content-Type: application/json
Authorization: Session (must be logged in)
```

**Request body:**
```json
{
  "analysis_type": "all",
  "x_data": [1, 2, 3, 4, 5],
  "y_data": [100, 150, 200, 260, 300],
  "discount_rate": 0.10,
  "cashflows": [-1000, 300, 400, 500, 600],
  "base_revenue": 1000000,
  "growth_mean": 0.08,
  "growth_std": 0.05
}
```

**Response:**
```json
{
  "success": true,
  "results": {
    "type": "all",
    "bisection": { "root": 25.0, "iterations": 14, "converged": true },
    "newton":    { "root": 25.0, "iterations": 4,  "converged": true },
    "regression": { "slope": 50.0, "intercept": 46.0, "r_squared": 0.9966 },
    "npv":       { "npv": 388.78, "decision": "Accept (NPV > 0)" },
    "irr":       18.45,
    "payback":   2.6
  }
}
```

---

## 9. Feature Walkthrough

### 9.1 Registration Flow

1. User visits `/register/`
2. Fills: First name, Last name, Email, Username, Password (×2), Company, Industry
3. Django's `UserCreationForm` validates password strength
4. `UserProfile` created alongside `User`
5. Auto-login → redirect to `/dashboard/`

### 9.2 Analysis Flow

1. Select analysis type from dropdown
2. Enter parameters (form fields update dynamically via JS)
3. Click **Run Analysis**
4. Server validates → dispatches to `numerical.py`
5. Results rendered with:
   - Stat cards (key numbers)
   - Iteration tables (bisection / Newton)
   - Matplotlib charts (base64 embedded PNG)
   - NPV waterfall / Monte Carlo fan chart
6. `AnalysisHistory` record saved

### 9.3 History Flow

1. `/history/` lists all past analyses in a table
2. Click **View** to open detail page
3. Detail page shows JSON input + result dumps

---

## 10. Testing Guide

### Unit Tests for numerical.py

```python
# predictor/tests.py
from django.test import TestCase
from .numerical import bisection, newton_raphson, profit, profit_derivative, npv, irr

class TestBisection(TestCase):
    def test_root_found(self):
        result = bisection(profit, 0, 50)
        self.assertAlmostEqual(result["root"], 25.0, places=2)

    def test_converged(self):
        result = bisection(profit, 0, 50)
        self.assertTrue(result["converged"])

class TestNewtonRaphson(TestCase):
    def test_root_found(self):
        result = newton_raphson(profit, profit_derivative, x0=10)
        self.assertAlmostEqual(result["root"], 25.0, places=2)

    def test_faster_than_bisection(self):
        bis = bisection(profit, 0, 50)
        nwt = newton_raphson(profit, profit_derivative, x0=10)
        self.assertLess(nwt["iterations"], bis["iterations"])

class TestNPV(TestCase):
    def test_positive_npv(self):
        result = npv(0.10, [-1000, 300, 400, 500, 600])
        self.assertGreater(result["npv"], 0)

    def test_zero_rate(self):
        # At 0% discount, NPV = sum of cashflows
        result = npv(0.0, [-1000, 300, 400, 500])
        self.assertAlmostEqual(result["npv"], 200.0, places=2)

class TestIRR(TestCase):
    def test_irr_makes_npv_zero(self):
        cashflows = [-1000, 300, 400, 500, 600]
        irr_rate = irr(cashflows) / 100
        npv_at_irr = npv(irr_rate, cashflows)["npv"]
        self.assertAlmostEqual(npv_at_irr, 0.0, places=2)
```

**Run tests:**
```bash
python manage.py test predictor
```

### Integration Tests

```python
class TestAnalysisView(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("testuser", password="testpass123")
        self.client.login(username="testuser", password="testpass123")

    def test_analysis_post(self):
        response = self.client.post("/analysis/", {
            "analysis_type": "breakeven",
            "analysis_name": "Test",
            "x_data_raw": "1,2,3,4,5",
            "y_data_raw": "100,150,200,260,300",
            "discount_rate": "0.10",
            "cashflows_raw": "-1000,300,400,500",
            "base_revenue": "1000000",
            "growth_mean": "0.08",
            "growth_std": "0.05",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("bisection", response.context["results"])

    def test_history_saved(self):
        # After analysis, check DB record was created
        initial_count = AnalysisHistory.objects.filter(user=self.user).count()
        self.test_analysis_post()
        self.assertEqual(
            AnalysisHistory.objects.filter(user=self.user).count(),
            initial_count + 1
        )
```

---

## 11. Deployment Guide

### Production Checklist

```python
# settings.py changes for production
DEBUG = False
SECRET_KEY = os.environ.get("SECRET_KEY")
ALLOWED_HOSTS = ["yourdomain.com", "www.yourdomain.com"]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("DB_NAME"),
        "USER": os.environ.get("DB_USER"),
        "PASSWORD": os.environ.get("DB_PASSWORD"),
        "HOST": os.environ.get("DB_HOST"),
        "PORT": "5432",
    }
}

# Static files
STATIC_ROOT = BASE_DIR / "staticfiles"
# Run: python manage.py collectstatic
```

### Deploy to Railway / Render / Heroku

```bash
# Add Procfile
echo "web: gunicorn fintech_project.wsgi" > Procfile

# Add gunicorn
pip install gunicorn psycopg2-binary
pip freeze > requirements.txt

# Push
git init && git add . && git commit -m "Initial commit"
# Connect to Railway/Render and deploy
```

---

## 12. Viva Q&A

**Q1: What is the Bisection Method and why does it work?**
The Bisection Method finds the root of f(x) = 0 by repeatedly halving an interval [a, b] where f(a) and f(b) have opposite signs. By the Intermediate Value Theorem, if f is continuous and f(a)·f(b) < 0, a root must exist between a and b. Each iteration eliminates half the interval, guaranteeing convergence.

**Q2: Why is Newton-Raphson faster than Bisection?**
Newton-Raphson has quadratic convergence — the number of correct decimal digits roughly doubles each iteration. Bisection has linear convergence — it gains only one bit of precision per iteration. For a tolerance of 10⁻⁴, Bisection may take 15 steps while Newton-Raphson takes 4–5.

**Q3: When would Newton-Raphson fail?**
It fails when: (a) f'(x) = 0 at some iterate (division by zero), (b) the starting guess is far from the root and the function is non-monotone, or (c) the function has discontinuities. Bisection is more robust for ill-conditioned functions.

**Q4: What does R² measure in regression?**
R² (coefficient of determination) measures the proportion of variance in y explained by x. R²=1 means the model fits perfectly; R²=0 means the regression line is no better than predicting the mean of y. For revenue forecasting, R² > 0.95 is considered excellent.

**Q5: What is NPV and how do you interpret it?**
NPV (Net Present Value) is the sum of all future cash flows discounted to the present using a required rate of return. A positive NPV means the investment creates value above the cost of capital — it should be accepted. A negative NPV means it destroys value — reject it.

**Q6: What is IRR and how is it computed?**
IRR (Internal Rate of Return) is the discount rate that makes NPV = 0. It is found by applying Newton-Raphson to the NPV function, which is a polynomial in (1+r). The IRR represents the investment's own rate of return; if IRR > hurdle rate, accept the investment.

**Q7: What is Monte Carlo simulation and why is it useful?**
Monte Carlo simulation models uncertainty by running thousands of random scenarios. Instead of a single revenue forecast, it produces a distribution of outcomes. The width of the percentile band shows risk — a wide band means high uncertainty; a narrow band means low uncertainty. This is far more informative than a point estimate.

**Q8: Why Django for this project?**
Django provides the ORM (database abstraction), routing, templating, CSRF protection, session management, and authentication system out of the box — allowing the project to focus on numerical methods rather than web infrastructure. It follows the MVT (Model-View-Template) pattern, which is clean and testable.

**Q9: How does the break-even analysis work mathematically?**
The profit function P(x) = 2x² - 100x + 1000 represents profit as a function of units sold. Break-even occurs at P(x) = 0. Since this is a quadratic, it has (generally) two roots. Analytically: x = (100 ± √(10000 - 8000)) / 4 = (100 ± √2000) / 4 ≈ 25.0 and 75.0. The numerical methods find these roots without needing the quadratic formula.

**Q10: What is the advantage of saving analyses to a database?**
Persistence allows users to track decisions over time, compare results across different parameters, and audit their analysis history. The JSONField stores arbitrary result structures without needing to define every column — making it flexible as new analysis types are added.

**Q11: What is exponential smoothing?**
Exponential smoothing is a forecasting technique: Sₜ = α·Yₜ + (1-α)·Sₜ₋₁. The parameter α (0 < α < 1) controls how quickly the model forgets old data. High α = more responsive to recent changes; low α = smoother but slower to react. It is useful for time series with no clear trend or seasonality.

**Q12: How would you scale this system to handle 10,000 users?**
Switch from SQLite to PostgreSQL. Add Redis for caching analysis results. Move Matplotlib chart generation to a Celery background task queue. Add nginx as a reverse proxy in front of gunicorn. Deploy on Docker/Kubernetes with horizontal scaling. Add a CDN for static assets.

---

## Final Statement (Viva Closing Line)

> *"FinSight Pro integrates six numerical methods — Bisection, Newton-Raphson, Linear Regression, NPV, IRR, and Monte Carlo Simulation — with a full-stack Django web application featuring authentication, persistent history, dynamic dark-themed charts, and a JSON API. It demonstrates that mathematical rigor and product-quality engineering are not mutually exclusive."*
