from django.db import models


class EsgAnalysis(models.Model):
    uploaded_file = models.FileField(upload_to="uploads/", blank=True, null=True)
    original_text = models.TextField()
    result_json   = models.JSONField()
    verdict       = models.CharField(max_length=20)
    score         = models.FloatField()
    confidence    = models.FloatField()
    created_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "ESG Analysis"
        verbose_name_plural = "ESG Analyses"

    def __str__(self):
        return f"[{self.verdict}] score={self.score:.2f}  ({self.created_at:%Y-%m-%d %H:%M})"
