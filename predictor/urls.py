"""
urls.py — URL Routing
FinSight Pro
"""

from django.urls import path
from . import views

urlpatterns = [
    path("",                views.landing,        name="landing"),
    path("dashboard/",      views.dashboard,      name="dashboard"),
    path("analysis/",       views.analysis_view,  name="analysis"),
    path("history/",        views.history_view,   name="history"),
    path("history/<int:pk>/", views.history_detail, name="history_detail"),
    path("register/",       views.register_view,  name="register"),
    path("login/",          views.login_view,     name="login"),
    path("logout/",         views.logout_view,    name="logout"),
    path("api/analyze/",    views.api_analyze,    name="api_analyze"),
]
