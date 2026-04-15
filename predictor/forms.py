"""
forms.py — Django Forms
FinSight Pro
"""

from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
import json


class RegisterForm(UserCreationForm):
    first_name = forms.CharField(max_length=50, required=True)
    last_name = forms.CharField(max_length=50, required=True)
    email = forms.EmailField(required=True)
    company = forms.CharField(max_length=100, required=False)
    industry = forms.CharField(max_length=100, required=False)

    class Meta:
        model = User
        fields = ("username", "first_name", "last_name", "email", "password1", "password2")

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        user.first_name = self.cleaned_data["first_name"]
        user.last_name = self.cleaned_data["last_name"]
        if commit:
            user.save()
        return user


class AnalysisForm(forms.Form):
    ANALYSIS_CHOICES = [
        ("breakeven", "Break-Even Analysis"),
        ("forecast", "Revenue Forecast"),
        ("npv", "NPV / IRR Analysis"),
        ("monte_carlo", "Monte Carlo Simulation"),
        ("all", "Full Suite (All Methods)"),
    ]

    analysis_type = forms.ChoiceField(choices=ANALYSIS_CHOICES)
    analysis_name = forms.CharField(max_length=100, required=False, initial="My Analysis")

    # Revenue data
    x_data_raw = forms.CharField(
        required=False,
        initial="1,2,3,4,5",
        help_text="Comma-separated period numbers"
    )
    y_data_raw = forms.CharField(
        required=False,
        initial="100,150,200,260,300",
        help_text="Comma-separated revenue values"
    )

    # NPV inputs
    discount_rate = forms.FloatField(required=False, initial=0.10, min_value=0.0, max_value=1.0)
    cashflows_raw = forms.CharField(
        required=False,
        initial="-1000,300,400,500,600",
        help_text="Comma-separated cash flows (negative = outflow)"
    )

    # Monte Carlo
    base_revenue = forms.FloatField(required=False, initial=1000000)
    growth_mean = forms.FloatField(required=False, initial=0.08)
    growth_std = forms.FloatField(required=False, initial=0.05)

    def clean(self):
        cleaned = super().clean()

        # Parse x_data
        x_raw = cleaned.get("x_data_raw", "1,2,3,4,5")
        try:
            cleaned["x_data"] = [float(v.strip()) for v in x_raw.split(",") if v.strip()]
        except ValueError:
            self.add_error("x_data_raw", "Must be comma-separated numbers.")

        # Parse y_data
        y_raw = cleaned.get("y_data_raw", "100,150,200,260,300")
        try:
            cleaned["y_data"] = [float(v.strip()) for v in y_raw.split(",") if v.strip()]
        except ValueError:
            self.add_error("y_data_raw", "Must be comma-separated numbers.")

        # Parse cashflows
        cf_raw = cleaned.get("cashflows_raw", "-1000,300,400,500,600")
        try:
            cleaned["cashflows"] = [float(v.strip()) for v in cf_raw.split(",") if v.strip()]
        except ValueError:
            self.add_error("cashflows_raw", "Must be comma-separated numbers.")

        return cleaned
