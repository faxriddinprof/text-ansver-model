from django.contrib import admin

from .models import EsgAnalysis


@admin.register(EsgAnalysis)
class EsgAnalysisAdmin(admin.ModelAdmin):
    list_display  = ("id", "verdict", "score", "threshold", "confidence", "created_at")
    list_filter   = ("verdict",)
    search_fields = ("original_text", "verdict")
    readonly_fields = (
        "uploaded_file", "original_text", "result_json",
        "verdict", "score", "threshold", "confidence", "created_at",
    )
    ordering = ("-created_at",)
