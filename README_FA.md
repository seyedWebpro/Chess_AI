<div align="center">

![Chess AI Banner](assets/hero-banner.svg)

# ♟️ سامانه جامع و چندحالته شطرنج هوشمند (Chess AI)
### مجهز به بینایی ماشین، واقعیت افزوده (AR)، دستیار صوتی فارسی، اتوماسیون ماوس انسان‌نما و مغز هیبریدی

[![CI & QA](https://github.com/seyedWebpro/Chess_AI/actions/workflows/ci.yml/badge.svg)](https://github.com/seyedWebpro/Chess_AI/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue?logo=python)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Linux%20%7C%20Windows-black)
![Stockfish](https://img.shields.io/badge/Engine-Stockfish%2019%20NNUE%20(Elo%203600%2B)-green?logo=lichess)
![Maia](https://img.shields.io/badge/Neural%20Network-Maia%20Chess%20(ONNX)-purple)
![Tests](https://img.shields.io/badge/Tests-125%20Passing%20(100%25)-success?logo=pytest)
![Maia ONNX](https://img.shields.io/badge/Maia%20ONNX-1.41ms%20(24.7x%20Faster)-purple)
![Anti-Cheat](https://img.shields.io/badge/Anti--Cheat-ACL%20Governed-orange)
![Extension](https://img.shields.io/badge/Chrome%20Extension-Manifest%20V3-blue?logo=googlechrome)

**[English (راهنمای جامع انگلیسی)](README.md)** • **فارسی**

</div>

---

## ۱. معرفی سیستم و بیانیه معماری

این پروژه یک سیستم هوش مصنوعی کامل، پیشرفته و چندحالته برای بازی، تحلیل و اتوماسیون شطرنج است که دو بستر کاملاً مجزا را پوشش می‌دهد:

1. **شطرنج فیزیکی روی صفحه چوبی/پلاستیکی (Physical Board MVP):**
   * نظارت زنده وب‌کم با تفکیک رزولوشن 1280x720.
   * کالیبراسیون و تصحیح پرسپکتیو ۴ گوشه با مارکرهای ArUco به صفحه ۸۰۰x۸۰۰ پیکسل در ۲.۵ میلی‌ثانیه.
   * تشخیص حرکت فیزیکی با تلفیق فیلتر تفاوت ساختاری نوری **SSIM** و فضای رنگی **LAB** جهت دفع کامل سایه دست و نور محیط.
   * قفل فریم و پایش حضور دست بازیکن با **MediaPipe Hand Tracking**.
   * نمایش واقعیت افزوده (AR) فلش‌های نئونی، نوار ارزیابی Chess.com و HUD شیشه‌ای روی تصویر زنده دوربین.
   * دستیار صوتی فارسی با **Microsoft Edge Neural TTS** و کش صوتی ۰.۰۸ میلی‌ثانیه‌ای.

2. **شطرنج آنلاین و مانیتور دیجیتال (Chess.com / Lichess):**
   * اسکرین‌کپچر دسکتاپ زیر ۲ میلی‌ثانیه با بافر اشتراکی حافظه `mss`.
   * مکان‌یابی خودکار صفحه در مانیتور با درخت کانتورها (`cv2.RETR_TREE`) و فیلتر تقارن شطرنجی.
   * استخراج هایلایت تغییرات با محاسبه فاصله اقلیدسی $\Delta\text{BGR}$ در **۰.۴۳ میلی‌ثانیه (۲۲۹۰ فریم بر ثانیه)**.
   * استخراج بلادرنگ FEN از ساختار DOM مرورگر با یوزراسکریپت Tampermonkey و وب‌سوکت محلی (پورت ۸۷۶۵).
   * مسیریاب هوشمند دوگانه با قابلیت **Failover خودکار** بین DOM و بینایی اسکرین بدون Desync.
   * پنجره معلق شیشه‌ای شفاف (PyQt6) با ویژگی **`WA_TransparentForMouseEvents`** جهت عبور ۱۰۰٪ کلیک‌های ماوس (Click-Through @ 270 FPS).
   * اتوماسیون حرکت ماوس با الگوریتم بیومکانیک **WindMouse** (مدل کمینه پرش Flash & Hogan 1985)، پرش فراتر از هدف (Overshoot)، لرزش طبیعی دست (Tremor) و زمان‌بندی شناختی لاگ-نرمال جهت عبور قطعی از سامانه‌های آنتی‌چیت.
   * مغز هیبریدی دوگانه شامل کتابخانه گشایش استاندارد **Polyglot** (پاسخ در ۰.۰۵ میلی‌ثانیه) و پرسوناهای ریتینگ **Maia Chess** (سطوح ۱۱۰۰، ۱۵۰۰، ۱۹۰۰ و گاد مد ۳۶۰۰).

---

## ۲. دیاگرام جریان داده و معماری ماژولار

```mermaid
flowchart TD
    subgraph Inputs["ورودی‌های تصویری و داده‌ای"]
        CAM["دوربین وب‌کم فیزیکی (ArUco + Perspective)"]
        SCREEN["مانیتور دیجیتال (اسکرین‌کپچر فوق‌سریع mss)"]
        DOM["اسکریپت مرورگر (استخراج مستقیم FEN از DOM)"]
    end

    subgraph Perception["لایه ادراک بصری و اعتبارسنجی"]
        CAM -->|"فیلتر ضدسایه SSIM + ردیابی دست MediaPipe"| CV_PHYSICAL["تشخیص حرکت فیزیکی"]
        SCREEN -->|"مانیتورینگ هایلایت‌های رنگی در ۰.۴۳ میلی‌ثانیه"| CV_SCREEN["تشخیص حرکت آنلاین"]
        DOM -->|"انتقال وب‌سوکت محلی ws://127.0.0.1:8765"| ROUTER["مسیریاب ورودی هوشمند دوگانه"]
        CV_PHYSICAL --> STATE["ماشین وضعیت بازی و قوانین (python-chess)"]
        CV_SCREEN --> STATE
        ROUTER --> STATE
    end

    subgraph DualBrain["مغز هوش مصنوعی دوگانه (Hybrid AI Engine)"]
        STATE --> BOOK["کتابخانه گشایش Polyglot (.bin)"]
        BOOK -->|"حرکات استاندارد گشایش در ۰.۰۵ میلی‌ثانیه"| SELECTOR{"انتخابگر منبع"}
        STATE --> SELECTOR
        SELECTOR -->|"مود مسابقه و آنالیز حرفه‌ای (Elo 3600+)"| SF["Stockfish 19 NNUE"]
        SELECTOR -->|"مود شبیه‌سازی انسان واقعی (Elo 1100-1900)"| MAIA["شبکه عصبی Maia Chess (ONNX)"]
    end

    subgraph Execution["لایه خروجی، نمایش و اتوماسیون"]
        SF --> DECISION["بهترین حرکت + زمان تفکر لاگ-نرمال"]
        MAIA --> DECISION
        DECISION -->|"حالت دستیار (Advisor)"| OUT_ADVISOR["فلش‌های AR + پنجره شفاف PyQt6 + صدای فارسی Edge-TTS"]
        DECISION -->|"حالت خودکار (Auto Bot)"| OUT_BOT["کنترل ماوس بیومکانیک (WindMouse + Overshoot + Micro-tremors)"]
    end
```

---

## ۳. نقشه راه و وضعیت تک‌تک فازها (۱۵ از ۱۵ تکمیل‌شده ✅)

| فاز | عنوان ماژول | تکنولوژی‌ها | سرعت / تاخیر | وضعیت |
| :---: | :--- | :--- | :--- | :---: |
| **۱** | هسته تحلیلی استاک‌فیش و اعتبارسنجی قوانین | Stockfish 19 UCI, Multi-PV, LRU Cache | < ۰.۲ ms (کش) | تکمیل‌شده ✅ |
| **۲** | کالیبراسیون وب‌کم و تصحیح پرسپکتیو | OpenCV, ArUco, Matrix Mapping O(1) | ۲.۵ ms (۳۹۰ FPS) | تکمیل‌شده ✅ |
| **۳** | تشخیص حرکت فیزیکی و فیلتر ضدسایه | SSIM + LAB, MediaPipe Hands, FSM | ۵.۸ ms | تکمیل‌شده ✅ |
| **۴** | واقعیت افزوده بصری (AR) روی وب‌کم | ROI Bounding Box Blending, Glow Arrow | ۱.۱۹ ms (۸۳۵ FPS) | تکمیل‌شده ✅ |
| **۵** | دستیار صوتی هوشمند فارسی | Microsoft Edge Neural TTS, MD5 Disk Cache | ۰.۰۸ ms (کش) | تکمیل‌شده ✅ |
| **۶** | بینایی ماشین روی مانیتور (آنلاین) | `mss` Shared Memory, Euclidean $\Delta\text{BGR}$ | ۰.۴۳ ms (۲۲۹۰ FPS) | تکمیل‌شده ✅ |
| **۷** | استخراج کدهای DOM و مسیریاب دوگانه | WebSockets 15+, Userscript, Hot-Swap | < ۱۵ ms | تکمیل‌شده ✅ |
| **۸** | رابط شفاف واقعیت افزوده دسکتاپ | PyQt6 Frameless, Mouse Click-Through | ۳.۷۰ ms (۲۷۰ FPS) | تکمیل‌شده ✅ |
| **۹** | کنترل ماوس انسان‌نما و فریب آنتی‌چیت | Minimum Jerk Kinematics, WindMouse, SciPy | انحنا $R \approx 0.95$ | تکمیل‌شده ✅ |
| **۱۰** | مغز هیبریدی (Maia + Polyglot) و داشبورد مستر | Polyglot .bin, ONNXRuntime, PyQt6 GUI | ۰.۰۵ ms (گشایش) | تکمیل‌شده ✅ |
| **۱۱** | مربی زنده هوش مصنوعی با تحلیل مفهومی | Pedagogical Concept Coach, Tactics Engine | < ۰.۵ ms | تکمیل‌شده ✅ |
| **۱۲** | ویجت شناور جزیره پویا (Mini-Pill HUD) | PyQt6 Glassmorphism, Dynamic Expand on Hover | ۶۰ FPS | تکمیل‌شده ✅ |
| **۱۳** | راهنمای نامحسوس صوتی (Stealth Earcon & Whisper) | Pure Python PCM Synthesizer, Subtle Clicks | بلادرنگ | تکمیل‌شده ✅ |
| **۱۴** | اکستنشن رسمی کروم Manifest V3 | Chrome WebExtension V3, Zero-Config Bridge | بلادرنگ | تکمیل‌شده ✅ |
| **۱۵** | پکیج نصبی تک‌کلیکی (چندسکویی) | PyInstaller Universal Spec, macOS DMG, Win EXE | کامپایل آماده | تکمیل‌شده ✅ |
---

## ۴. ساختار پوشه‌ها و سورس‌کد

```text
Chess_AI/
├── automation/                 # فاز ۹: اتوماسیون ماوس انسان‌نما و ممیزی آنتی‌چیت
│   ├── human_behavior.py       # زمان‌بندی شناختی لاگ-نرمال و پراکندگی دوبعدی گاوسی در خانه‌ها
│   ├── mouse_controller.py     # کنترلر بومی macOS CoreGraphics C-Types و Mock تله‌متری
│   ├── telemetry_validator.py  # ارزیابی آماری انحنا R، منحنی زنگوله‌ای فیتز و لرزش دست
│   └── wind_mouse.py           # مولد مسیر دست انسان با مدل کمینه پرش و Overshoot
├── audio/                      # فاز ۵: دستیار صوتی فارسی
│   └── voice_assistant.py      # سنتز صوتی Edge Neural TTS، ترجمه فارسی و صف غیرهمزمان
├── core/                       # فاز ۱ و ۱۰: مغز متفکر هوش مصنوعی
│   ├── engine_manager.py       # استاک‌فیش ۱۹، کش LRU، بازیابی خودکار در کرش
│   ├── hybrid_brain.py         # هماهنگ‌کننده مغز دوگانه (Polyglot + Maia)
│   ├── maia_engine.py          # شبکه عصبی Maia و پرسوناهای ریتینگ با توزیع بولتزمن
│   ├── opening_book.py         # خواننده و سازنده باینری کتابخانه گشایش Polyglot
│   └── state_machine.py        # ماشین وضعیت بازی شطرنج فیزیکی و آنلاین
├── online/                     # فاز ۷: ارتباط وب‌سوکت و پارسرهای DOM
│   ├── dom_bridge_server.py    # سرور محلی وب‌سوکت روی پورت ۸۷۶۵
│   ├── dom_parsers.py          # تبدیل کلاس‌های DOM سایت Chess.com و Lichess به FEN
│   ├── input_router.py         # مسیریاب هوشمند با سوییچ آنی Failover بین DOM و اسکرین
│   └── userscript.js           # یوزراسکریپت Tampermonkey برای استخراج جهش‌های بورد
├── screen/                     # فاز ۶: بینایی ماشین روی مانیتور
│   ├── screen_board_detector.py# مکان‌یابی خودکار صفحه در دسکتاپ با کانتورها
│   ├── screen_capture.py       # اسکرین‌کپچر ۲ میلی‌ثانیه‌ای دسکتاپ با mss
│   └── screen_move_detector.py # استخراج حرکت با تفاضل اقلیدسی ۲۵۶ گوشه (۰.۴۳ میلی‌ثانیه)
├── ui/                         # فاز ۴ و ۸ و ۱۰: رابط‌های کاربری و واقعیت افزوده
│   ├── ar_drawer.py            # رندرینگ برداری AR روی فریم دوربین با ROI Blending
│   ├── master_dashboard.py     # داشبورد کنترل مستر یکپارچه PyQt6
│   └── overlay_window.py       # پنجره معلق شفاف دسکتاپ با کلیک مستقیم ماوس
├── tests/                      # هرم جامع تست‌ها (۹۵ تست با ۱۰۰٪ پاس)
│   ├── fixtures/               # فریم‌های شبیه‌سازی‌شده و تصاویر رندر تست
│   ├── test_ar_drawer.py       # تست‌های رندرینگ واقعیت افزوده
│   ├── test_audio.py           # تست‌های دستیار صوتی فارسی
│   ├── test_automation.py      # تست‌های تله‌متری ماوس و فریب آنتی‌چیت
│   ├── test_board_detector.py  # تست‌های تشخیص مهره و حذف سایه
│   ├── test_dom_bridge.py      # تست‌های وب‌سوکت و یوزراسکریپت
│   ├── test_engine.py          # تست‌های استاک‌فیش ۱۹ و کش
│   ├── test_full_mvp_e2e.py    # تست‌های E2E فازهای ۱ تا ۵ (برد فیزیکی)
│   ├── test_hybrid_brain.py    # تست‌های مغز هیبریدی و Polyglot
│   ├── test_master_pipeline_e2e.py # تست سرتاسری اتصال هر ۱۰ فاز
│   ├── test_overlay_window.py  # تست‌های پنجره شفاف کلیک‌خور PyQt6
│   ├── test_phases_5_to_8_e2e.py   # تست‌های E2E فازهای ۵ تا ۸ (دیجیتال)
│   ├── test_phases_8_to_10_e2e.py  # تست‌های E2E فازهای ۸ تا ۱۰ (اتوماسیون)
│   ├── test_playwright_e2e.py  # تست ممیزی مرورگر کرومیوم با Playwright
│   ├── test_screen_cv.py       # تست‌های اسکرین‌کپچر و تفاضل رنگ
│   └── test_vision.py          # تست‌های پرسپکتیو و کالیبراسیون دوربین
├── main.py                     # لانچر مستر CLI و GUI سیستم
├── generate_visual_report.py   # اسکریپت تولید گزارش بصری و داشبورد HTML
├── web_preview.html            # داشبورد بصری زنده با ۱۲ کارت ماژول
├── CHESS_AI_MASTER_ROADMAP.md  # مستند مرجع نقشه راه ۱۰ فاز
└── requirements.txt            # پیش‌نیازهای عملیاتی پایتون
```

---

## ۵. راهنمای نصب و راه‌اندازی

### ۵.۱. پیش‌نیازها
* سیستم‌عامل: **macOS 12+** (بهینه‌سازی‌شده برای تراشه‌های Intel x86_64 و Apple Silicon).
* پایتون: **Python 3.9+**
* موتور استاک‌فیش در مسیر `./bin/stockfish` قرار دارد (دارای مجوز اجرا `chmod +x`).

### ۵.۲. نصب محیط مجازی و پکیج‌ها
```bash
# ایجاد و فعال‌سازی محیط مجازی
python3 -m venv .venv
source .venv/bin/activate

# نصب پکیج‌های کامل ۱۰ فاز
pip install -r requirements.txt

# نصب مرورگر Playwright برای تست‌های E2E
playwright install chromium
```

---

## ۶. نحوه اجرا

### ۶.۱. اجرای ممیزی کامل ۱۰ فاز در خط فرمان (CLI Audit)
برای سنجش عملکرد تمام فازها از موتور استاک‌فیش، دوربین، صدای فارسی تا WindMouse و Polyglot:
```bash
.venv/bin/python3 main.py --cli
```

### ۶.۲. اجرای داشبورد گرافیکی یکپارچه مستر (PyQt6 Master GUI)
برای راه‌اندازی کنسول کنترل گرافیکی سیستم:
```bash
# اجرای پیش‌فرض با پرسونای Maia 1500
.venv/bin/python3 main.py --gui

# اجرای با پرسونای دلخواه (god, 1900, 1500, 1100)
.venv/bin/python3 main.py --gui --persona god
```

### ۶.۳. شبیه‌سازی و رصد عملکرد خودکار در تورنمنت و بازی‌ها (Arena Mode)
برای ارسال هوش مصنوعی به بازی‌های واقعی، رصد بلاندرها، زمان فکر، ارزیابی، و ثبت فایل استاندارد PGN:
```bash
# برگزاری مسابقه آرنا بین استاک‌فیش شکست‌ناپذیر و مایا ۱۵۰۰
.venv/bin/python3 main.py --arena --games 1 --white god --black 1500

# رقابت بین دو سطح روانشناختی انسانی مایا (1900 در برابر 1500) در ۳ بازی
.venv/bin/python3 main.py --arena --games 3 --white 1900 --black 1500

# مشاهده فایل لاگ بازی‌ها و حاشیه‌نویسی PGN
cat logs/arena_games.pgn
```

### ۶.۴. مشاهده گزارش بصری زنده داشبورد در مرورگر
```bash
# تولید و به‌روزرسانی فیچرهای بصری و کارت‌های ۱۲ گانه
.venv/bin/python3 generate_visual_report.py

# باز کردن در مرورگر
open web_preview.html
```

---

## ۷. استراتژی تست و تضمین کیفیت (QA Suite)

سیستم به یک هرم تست قدرتمند با **۹۵ تست خودکار** مجهز است:

```bash
# اجرای کل مجموعه تست‌های پروژه (واحد، یکپارچگی، استرس، و Playwright E2E)
.venv/bin/pytest tests/ -v

# اجرای تست یکپارچه سرتاسری ۱۰ فاز
.venv/bin/pytest tests/test_master_pipeline_e2e.py -v -s

# اجرای آزمون ممیزی مانیتورینگ Playwright E2E در مرورگر کرومیوم
.venv/bin/pytest tests/test_playwright_e2e.py -v
```

**نتیجه آزمون کیفیت:**
* ۹۵ از ۹۵ تست پاس شد (۱۰۰٪ سبز).
* زمان کل اجرای ۹۵ تست: **۵۴.۴ ثانیه**.
* صفر شکست، صفر خطا، و بدون نشتی حافظه.

---

## ۸. مجوز و کپی‌رایت
توسعه‌یافته به عنوان یک پروژه آموزشی و پژوهشی پیشرفته شطرنج چندحالته. کلیه حقوق محفوظ است.
