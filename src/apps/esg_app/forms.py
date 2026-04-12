from django import forms


class UploadForm(forms.Form):
    file = forms.FileField(
        label="Upload .txt project document",
        help_text="Only plain-text (.txt) files are accepted.",
        widget=forms.ClearableFileInput(attrs={"accept": ".txt"}),
    )

    def clean_file(self):
        f = self.cleaned_data["file"]
        if not f.name.endswith(".txt"):
            raise forms.ValidationError("Only .txt files are allowed.")
        if f.size > 5 * 1024 * 1024:  # 5 MB guard
            raise forms.ValidationError("File size must not exceed 5 MB.")
        return f
