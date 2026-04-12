from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("esg_app", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="esganalysis",
            name="threshold",
            field=models.FloatField(default=3.0),
        ),
        migrations.AlterField(
            model_name="esganalysis",
            name="confidence",
            field=models.IntegerField(default=50),
        ),
    ]
