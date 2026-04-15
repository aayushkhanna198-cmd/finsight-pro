"""
models.py — Database Models
FinSight Pro
"""

from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class AnalysisHistory(models.Model):
    """Stores past analysis runs for authenticated users."""

    ANALYSIS_TYPES = [
        ("breakeven", "Break-Even Analysis"),
        ("forecast", "Revenue Forecast"),
        ("npv", "NPV Analysis"),
        ("monte_carlo", "Monte Carlo Simulation"),
        ("irr", "IRR Calculation"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="analyses")
    analysis_type = models.CharField(max_length=20, choices=ANALYSIS_TYPES)
    name = models.CharField(max_length=100, default="Untitled Analysis")
    input_data = models.JSONField()
    result_data = models.JSONField()
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]
        verbose_name_plural = "Analysis Histories"

    def __str__(self):
        return f"{self.user.username} — {self.get_analysis_type_display()} ({self.created_at.strftime('%Y-%m-%d')})"


class UserProfile(models.Model):
    """Extended profile for each user."""
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    company = models.CharField(max_length=100, blank=True)
    industry = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Profile of {self.user.username}"
