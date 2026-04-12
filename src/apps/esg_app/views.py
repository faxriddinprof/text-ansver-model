import json

from django.shortcuts import render, get_object_or_404
from django.views.decorators.http import require_http_methods

from .forms import UploadForm
from .models import EsgAnalysis
from .pipeline import read_file_content, run_esg_pipeline, parse_result


@require_http_methods(["GET", "POST"])
def analyze(request):
    form    = UploadForm()
    context = {"form": form, "result": None, "error": None}

    if request.method != "POST":
        return render(request, "esg_app/analyze.html", context)

    form = UploadForm(request.POST, request.FILES)
    context["form"] = form

    if not form.is_valid():
        return render(request, "esg_app/analyze.html", context)

    uploaded = form.cleaned_data["file"]

    try:
        text       = read_file_content(uploaded)
        raw_result = run_esg_pipeline(text)
        parsed     = parse_result(raw_result)
    except Exception as exc:  # engine errors must not crash the page
        context["error"] = f"Evaluation failed: {exc}"
        return render(request, "esg_app/analyze.html", context)

    # Reset file pointer before saving so Django can write it to disk
    uploaded.seek(0)

    record = EsgAnalysis.objects.create(
        uploaded_file = uploaded,
        original_text = text,
        result_json   = raw_result,
        verdict       = parsed["verdict"],
        score         = parsed["score"],
        threshold     = parsed["threshold"],
        confidence    = parsed["confidence_pct"],
    )

    context["result"]    = parsed
    context["record_id"] = record.pk
    context["raw_json"]  = json.dumps(raw_result, ensure_ascii=False, indent=2)
    return render(request, "esg_app/analyze.html", context)


@require_http_methods(["GET"])
def history(request):
    records = EsgAnalysis.objects.all()[:100]   # latest 100
    return render(request, "esg_app/history.html", {"records": records})


@require_http_methods(["GET"])
def detail(request, pk):
    record = get_object_or_404(EsgAnalysis, pk=pk)
    parsed   = parse_result(record.result_json)
    raw_json = json.dumps(record.result_json, ensure_ascii=False, indent=2)
    return render(request, "esg_app/detail.html", {
        "record":   record,
        "result":   parsed,
        "raw_json": raw_json,
    })
