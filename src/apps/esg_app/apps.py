from django.apps import AppConfig


class EsgAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.apps.esg_app"
    label = "esg_app"
    verbose_name = "ESG Evaluator"
