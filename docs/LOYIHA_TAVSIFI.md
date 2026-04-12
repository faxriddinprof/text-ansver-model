# Yashil Loyiha Baholash Tizimi — To'liq Tavsif

> **Versiya:** 2.0 (LLM + Validatsiya qatlami)  
> **Til:** Python 3.13 / Django 6  
> **LLM:** qwen2.5:7b via Ollama (`http://localhost:11434`)  
> **Test natijalari:** 8/8 = 100% aniqlik

---

## 1. Loyiha Maqsadi

Bu tizim O'zbekiston Respublikasining **"286-son Buyruq"** (Yashil loyihalar mezonlari) asosida taqdim etilgan loyiha hujjatlarini avtomatik ravishda tahlil qilib, loyiha **"YASHIL"** yoki **"YASHIL EMAS"** ekanligini aniqlaydi.

Tizim ikki turdagi hujjatlarni qabul qiladi:
- **`.txt`** — erkin matn ko'rinishidagi loyiha tavsifi (biznes-reja, texnik tavsif va hokazo)
- **`.json`** — tuzilgan ESG ma'lumotlar formati

---

## 2. Umumiy Arxitektura

```
Kirish hujjat
     │
     ▼
┌────────────────────────┐
│   parser.py            │  Matnni o'qish va tozalash
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│   extractor.py         │  LLM (qwen2.5:7b) — FAQAT faktlarni ajratib oladi
│                        │  → stop_factors (to'xtatuvchi omillar)
│                        │  → green_criteria (yashil mezonlar)
│                        │
│   Validatsiya qatlami  │  Python — LLM da'volarini kalit so'zlar bilan tekshiradi
│   (_validate_esg_response)│  Yolg'on ijobiy natijalarni rad etadi
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│   run_tests.py /       │  Python — FAQAT Python qaror qabul qiladi
│   Qaror mantiq         │  stop_factor=True → YASHIL EMAS
│                        │  score ≥ 3 → YASHIL
│                        │  else → YASHIL EMAS
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│   engine.py            │  (Zaxira rejim — LLM mavjud bo'lmaganda)
│                        │  Kalit so'z asosidagi qoidalar mexanizmi
└────────────────────────┘
```

**Asosiy tamoyil:** LLM faqat ma'lumot ajratadi, qarorni esa Python qabul qiladi.

---

## 3. Fayl Strukturasi

```
text-ansver-model/
│
├── manage.py                      # Django boshqaruv fayli
├── run_tests.py                   # Asosiy test ishga tushiruvchi va pipeline orchestrator
│
├── src/
│   └── utils/
│       ├── parser.py              # Matn o'qish va tozalash
│       ├── extractor.py           # LLM mijozi + validatsiya qatlami (asosiy modul)
│       ├── engine.py              # Qoidalar mexanizmi (zaxira rejim)
│       ├── rules.py               # Qoidalar yordamchi funksiyalari
│       ├── report.py              # Hisobot generatori
│       └── green_rules.json       # 286-son Buyruq: 41 qoida, 17 istisno, 8 bog'liq qoida
│
├── checks/
│   ├── simple.txt                 # Test: oddiy yashil loyiha → GREEN
│   ├── bad.txt                    # Test: neft/gaz loyihasi → NOT GREEN
│   ├── medium.txt                 # Test: o'rtacha loyiha → NOT GREEN
│   ├── tricky.txt                 # Test: murakkab loyiha → NOT GREEN
│   ├── big_test.txt               # Test: katta matnli loyiha → GREEN
│   ├── dependent.txt              # Test: bog'liq qoidalar → NOT GREEN
│   ├── real.txt                   # Test: haqiqiy bank hujjati → NOT GREEN
│   └── expected_results.json      # Har bir test uchun kutilgan natija
│
├── docs/
│   └── LOYIHA_TAVSIFI.md          # (siz hozir o'qiyotgan fayl)
│
├── config/                        # Django settings
├── static/                        # Statik fayllar
├── logs/                          # Jurnal fayllar
├── requirements.txt               # Python kutubxonalari
├── Makefile                       # Qulay buyruqlar
├── .env.example                   # Muhit o'zgaruvchilari namunasi
├── yashil-nizom.pdf               # 286-son Buyruq asl hujjati
└── ESG-hulosa.pdf                 # ESG mezonlari xulosasi
```

---

## 4. Asosiy Komponentlar

### 4.1 `parser.py` — Matn O'quvchi

Hujjat faylini o'qib, matnni tozalab qaytaradi.

```python
from src.utils.parser import read_txt
text = read_txt("checks/simple.txt")
```

Vazifalari:
- Faylni UTF-8 kodlashda o'qish
- Ortiqcha bo'sh joylarni tozalash
- Matnni keyingi qayta ishlash uchun tayyorlash

---

### 4.2 `extractor.py` — LLM Mijozi va Validatsiya Qatlami

Bu loyihaning eng muhim moduli. Unda ikki qism mavjud:

#### A) LLM orqali faktlarni ajratib olish

`analyze_esg_holistic(text)` funksiyasi:

1. Matnning dastlabki **6000 ta belgisi** LLM ga yuboriladi (loyihaning mohiyati har doim boshida bo'ladi)
2. LLM ga `ESG_ANALYST_PROMPT` ko'rsatmasi beriladi
3. LLM quyidagi JSON formatda javob qaytaradi:

```json
{
  "stop_factors": {
    "coal":     {"value": false, "evidence": ""},
    "oil_gas":  {"value": false, "evidence": ""},
    "alcohol":  {"value": false, "evidence": ""},
    "tobacco":  {"value": false, "evidence": ""},
    "gambling": {"value": false, "evidence": ""}
  },
  "green_criteria": {
    "renewable_energy":             {"value": false, "evidence": ""},
    "energy_efficiency":            {"value": false, "evidence": ""},
    "ghg_reduction":                {"value": false, "evidence": ""},
    "environmental_infrastructure": {"value": false, "evidence": ""},
    "certificate":                  {"value": false, "evidence": ""}
  }
}
```

> **Muhim:** LLM faqat `value: true/false` va `evidence` (dalil) qaytaradi.  
> LLM **hech qachon** "GREEN" / "NOT GREEN" qarorini bermaydi.

#### B) Validatsiya qatlami

`_validate_esg_response(raw_esg, text)` — LLM da'volarini Python kalit so'zlari bilan tasdiqlaydi:

| Manba | Mas'uliyat |
|-------|-----------|
| LLM | `value: true` taklif qiladi |
| Python `_validate_esg_response` | Kalit so'zlar yo'q bo'lsa `true` → `false` ga o'zgartiradi |
| Python `run_txt_pipeline` | `stop_factor` va `score` asosida qaror qabul qiladi |

**Kalit so'zlar bo'yicha mezonlar:**

| Mezon | Tasdiqlovchi kalit so'zlar (namunalar) |
|-------|---------------------------------------|
| `renewable_energy` | quyosh, solar, shamol, wind, gidroelektro, vge |
| `energy_efficiency` | energiya tejash, %20 kamaytirish, iso 50001 |
| `ghg_reduction` | co2, karbon, issiqxona gazi, ghg, emission |
| `environmental_infrastructure` | suv tozalash, chiqit qayta ishlash, filtr qurilmasi |
| `certificate` | leed, edge, breeam, ekologik sertifikat |
| `coal` | ko'mir, ugledobycha, coal mining |
| `oil_gas` | neft, gaz, neftni qayta ishlash |
| `alcohol` | spirt, alkogol, vino zavodi, pivo |
| `tobacco` | tamaki, sigaret, tutun |
| `gambling` | kazino, qimor, tikish |

---

### 4.3 `engine.py` — Qoidalar Mexanizmi (Zaxira Rejim)

LLM mavjud bo'lmaganda ishlatiladi.

`evaluate(data, rules_json, mode)` funksiyasi:
- `green_rules.json` dagi 41 ta qoidani tekshiradi
- 17 ta istisnoni qayta ishlaydi
- 8 ta bog'liq qoidani ko'rib chiqadi
- `"GREEN"` yoki `"NOT GREEN"` qaytaradi

`evaluate_from_esg_json(data)` funksiyasi:
- `.json` formatdagi tuzilgan ESG hujjatlarni qayta ishlaydi
- JSON pipeline uchun asosiy funksiya

---

### 4.4 `run_tests.py` — Pipeline Orchestrator

Uch xil pipeline mavjud:

| Pipeline | Ishlatilash holati | Aniqlik |
|----------|-------------------|---------|
| `txt_esg` | LLM mavjud, `.txt` fayl | Eng yuqori |
| `txt` | LLM mavjud emas, `.txt` fayl | O'rtacha |
| `esg_json` | `.json` fayl | Yuqori |

**TXT pipeline qaror mantiq (`run_txt_pipeline`):**

```python
# 1. LLM → faktlarni ajratib olish
esg = analyze_esg_holistic(text)

# 2. To'xtatuvchi omillar tekshiruvi
stop_triggered = any(v["value"] is True for v in stop_facs.values())

# 3. Yashil mezonlar soni
score = sum(1 for v in criteria.values() if v["value"] is True)

# 4. Python qaror qabul qiladi (LLM emas!)
if stop_triggered:
    final_status = "NOT GREEN"   # Neft/gaz/alkogol → avomatik rad
elif score >= 3:
    final_status = "GREEN"       # 5 dan 3 ta mezon bajarilgan
else:
    final_status = "NOT GREEN"   # Yetarli mezon yo'q
```

---

## 5. Qaror Qabul Qilish Mantiq

### 5.1 To'xtatuvchi Omillar (Stop Factors)

Quyidagi omillardan biri `true` bo'lsa, loyiha avtomatik **YASHIL EMAS**:

| Omil | Tavsif |
|------|--------|
| `coal` | Asosiy faoliyat: ko'mir qazib olish yoki qayta ishlash |
| `oil_gas` | Asosiy faoliyat: neft/gaz qazib olish yoki rafinirlash |
| `alcohol` | Asosiy faoliyat: alkogol ishlab chiqarish |
| `tobacco` | Asosiy faoliyat: tamaki mahsulotlari |
| `gambling` | Asosiy faoliyat: qimor yoki kazino |

### 5.2 Yashil Mezonlar (Green Criteria)

5 ta mezondan **3 yoki undan ko'pi** bajarilishi kerak:

| Mezon | Tavsif |
|-------|--------|
| `renewable_energy` | Quyosh/shamol/gidro energiya asosiy manba sifatida |
| `energy_efficiency` | Energiyadan foydalanish samaradorligi ≥20% yaxshilangan |
| `ghg_reduction` | CO₂ / issiqxona gazlari kamaytirish maqsadi aniq ko'rsatilgan |
| `environmental_infrastructure` | Suv tozalash, chiqit qayta ishlash, sanoat ifloslantirish nazorati |
| `certificate` | EDGE / LEED / BREEAM yoki rasmiy ekologik sertifikat |

### 5.3 Ishonch Darajasi (Confidence)

Python o'zi hisoblaydi (LLM emas):

```
confidence = 50
           + 10 × (bajarilgan mezonlar soni)
           + 5  × (kuchli signal: sertifikat yoki gidro)
           - 15 × (rad etilgan LLM da'volari soni)

Chegara: 0 – 100
```

---

## 6. Test Tizimi

### 6.1 Test Fayllar va Kutilgan Natijalar

| Fayl | Kutilgan | Izoh |
|------|---------|------|
| `simple.txt` | GREEN | Sodda yashil loyiha |
| `bad.txt` | NOT GREEN | Neft/gaz (stop factor: oil_gas) |
| `medium.txt` | NOT GREEN | O'rtacha, yetarli mezon yo'q |
| `tricky.txt` | NOT GREEN | Murakkab, chalgituvchi matn |
| `big_test.txt` | GREEN | Katta hajmli yashil loyiha |
| `dependent.txt` | NOT GREEN | Bog'liq qoidalar, yetarli emas |
| `real.txt` | NOT GREEN | Haqiqiy bank hujjati, yashil emas |
| `checks/real/fazo_group_esg.json` | NOT GREEN | JSON format, yashil emas |

### 6.2 Test Ishga Tushirish

```bash
# Barcha testlar (verbose rejim)
python3 run_tests.py

# Barcha testlar (toza rejim)
python3 run_tests.py --quiet

# Qattiq rejim (ko'proq salbiy holat tekshiruvi)
python3 run_tests.py --mode strict

# Bitta fayl
python3 run_tests.py --file checks/simple.txt

# Bitta fayl, toza rejim
python3 run_tests.py --quiet --file checks/bad.txt
```

### 6.3 Test Natijasi Formati

```
[TEST] simple.txt
  Pipeline : [txt_esg]
  Expected : GREEN
  Actual   : GREEN  (score: 3  confidence: 85%)
  Reason   : ...
  Result   : ✅ PASS

[TEST] bad.txt
  Pipeline : [txt_esg]
  Expected : NOT GREEN
  Actual   : NOT GREEN  (score: 0  confidence: 35%)
  Corrected: oil_gas claim overridden — no keyword match (original: true)
  [audit]  : LLM claimed oil_gas but no domain keywords found in text
  Result   : ✅ PASS
```

**`Corrected:`** — Validatsiya qatlami LLM da'vosini rad etdi  
**`[audit]:`** — Nima uchun rad etilgani haqida tushuntirish

---

## 7. Muammolar va Yechimlar

### 7.1 LLM Gallyutsinatsiyasi

**Muammo:** LLM meva sharbati zavodi hujjatida "alcohol production" deb yozdi.

**Yechim:** `_validate_esg_response()`:
- LLM `alcohol: true` taklif qilsa
- Python matinda `spirt`, `alkogol`, `vino`, `pivo` va boshqa kalit so'zlarni qidiradi
- Topilmasa → `true` → `false` ga o'zgartiriladi
- `Corrected:` satrida foydalanuvchiga ko'rsatiladi

### 7.2 Noto'g'ri Ijobiy Natijalar (False Positives)

**Muammo:** Bank hujjatidagi `filtr`, `suv`, `sertifikat` so'zlari tizimni "yashil" deb yanglitmoqda edi.

**Yechim:** 
- Faqat hujjatning **dastlabki 6000 ta belgisi** LLM ga yuboriladi (loyiha mohiyati hujjat boshida)
- `certificate` mezoni uchun faqat `leed`, `edge`, `breeam`, `ekologik sertifikat` kabi maxsus so'zlar hisobga olinadi
- Oddiy `sertifikat` so'zi yetarli emas

### 7.3 LLM Ishlamay Qolsa

**Yechim:** Avtomatik zaxira rejim:
- Katta fayllar (>20 000 belgi): `_extract_with_keywords(mode="strict")` + `evaluate()`
- Kichik fayllar: `extract_data(mode=MODE)` + `evaluate()`
- Pipeline: `txt_esg` → `txt` (avtomatik)

---

## 8. Texnik Talablar

### Muhit O'rnatish

```bash
# Virtual muhit yaratish
python3 -m venv .venv
source .venv/bin/activate

# Kutubxonalarni o'rnatish
pip install -r requirements.txt

# Ollama o'rnatish (macOS)
brew install ollama

# Modelni yuklab olish
ollama pull qwen2.5:7b

# Ollama serverini ishga tushirish
ollama serve
```

### `.env` Fayl

```env
DEBUG=True
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=qwen2.5:7b
```

### Asosiy Kutubxonalar

| Kutubxona | Maqsad |
|-----------|--------|
| `django` | Web framework |
| `requests` | Ollama HTTP API chaqiruvi |
| `python-dotenv` | Muhit o'zgaruvchilarini yuklash |

---

## 9. Tizim Ish Jarayoni (Diagramma)

```
Foydalanuvchi hujjat beradi
         │
         ▼
    .json fayl?
    ┌───┴───┐
   Ha      Yo'q
    │         │
    ▼         ▼
JSON pipeline  Ollama ishlayaptimi?
(esg_json)    ┌──────┴──────┐
              Ha           Yo'q
              │               │
              ▼               ▼
      analyze_esg_holistic  Kalit so'z
      (text[:6000] → LLM)   fallback
              │               │
              ▼               │
      _validate_esg_response  │
      (Python tekshiruvi)     │
              │               │
              └───────┬───────┘
                      │
                      ▼
              Python qaror:
              stop_factor → NOT GREEN
              score≥3 → GREEN
              else → NOT GREEN
                      │
                      ▼
              Natija + Ishonch darajasi
              + Rad etilgan da'volar
              + Audit yozuvlari
```

---

## 10. Natijalar Interpretatsiyasi

| Holat | Ma'nosi |
|-------|---------|
| `GREEN` | Loyiha 286-son Buyruq mezonlariga javob beradi |
| `NOT GREEN` | Loyiha yashil mezonlarga mos emas yoki to'xtatuvchi omil mavjud |
| `score: 3/5` | 5 ta mezondan 3 tasi bajarilgan |
| `confidence: 85%` | Tizim qaroriga 85% ishonch darajasi |
| `Corrected: ...` | LLM da'vosi validatsiya orqali to'g'irlandi |
| `[audit]: ...` | To'g'irlash sababi |

---

## 11. Kengaytirish Yo'llari

- **Yangi stop factor qo'shish:** `_STOP_FACTOR_KEYWORDS` ga yangi kalit so'z ro'yxati qo'shing va `ESG_ANALYST_PROMPT` ni yangilang
- **Yangi yashil mezon qo'shish:** `_GREEN_CRITERION_KEYWORDS` ga qo'shing, `ESG_ANALYST_PROMPT` va JSON sxemasini yangilang, qaror mantiqidagi `score >= 3` chegarasini ko'rib chiqing
- **Boshqa LLM ishlatish:** `extractor.py` dagi `OLLAMA_MODEL` va `OLLAMA_URL` ni o'zgartiring
- **Score chegarasini o'zgartirish:** `run_txt_pipeline()` dagi `score >= 3` ni `score >= 2` yoki `score >= 4` ga o'zgartiring

---

*Hujjat oxirgi marta yangilangan: 2025 | Loyiha: text-ansver-model v2.0*
