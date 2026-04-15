from django.contrib import admin
from .models import AnalysisHistory, UserProfile

@admin.register(AnalysisHistory)
class AnalysisHistoryAdmin(admin.ModelAdmin):
    list_display = ("user", "name", "analysis_type", "created_at")
    list_filter = ("analysis_type",)
    search_fields = ("user__username", "name")

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "company", "industry", "created_at")
