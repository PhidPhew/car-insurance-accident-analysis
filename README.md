# Weather-Aware Road Risk Intelligence for Motor Insurance

โปรเจกต์กลุ่มวิชา Data Mining (69-1) — Group Assignment #1: Mini Data Warehouse & Analytics Dashboard
คณะวิศวกรรมศาสตร์ มหาวิทยาลัยเกษตรศาสตร์

## เกี่ยวกับโปรเจกต์

ทีมเราสมมติบทบาทเป็นบริษัทประกันภัยรถยนต์ **"SiamShield Insurance"** ซึ่งปัจจุบันตั้งราคาเบี้ยประกัน
แบบหยาบ ๆ (แบ่งแค่กรุงเทพฯ vs. ต่างจังหวัด + ประเภทรถ) ทำให้เกิดปัญหา **adverse selection**:
พื้นที่เสี่ยงต่ำจ่ายแพงเกินจริงแล้วเลิกซื้อ ส่วนพื้นที่เสี่ยงสูงจ่ายถูกเกินจริงแล้วแห่มาซื้อ →
ขาดทุนจากการรับประกันภัย

เป้าหมายของระบบคือสร้าง pipeline แบบ end-to-end `raw data → ETL → star-schema DW → ML → dashboard`
เพื่อตอบคำถามว่าความเสี่ยงที่แท้จริงแปรผันตาม **พื้นที่ × เวลา × ประเภทรถ × สภาพอากาศ** อย่างไร
พร้อมพยากรณ์ล่วงหน้า 6 เดือน

รายละเอียดทั้งหมด (กติกาการทำงาน, ปัญหาข้อมูล 17 ข้อ, ออกแบบ warehouse, ML stack ฯลฯ) อยู่ใน
[`CLAUDE.md`](CLAUDE.md) — ไฟล์นี้คือ source of truth ของโปรเจกต์ ทุกคนควรอ่านก่อนเริ่มงาน

## โครงสร้างโฟลเดอร์

```
├── config/              ← path ทั้งหมดของโปรเจกต์ (settings.py), province master, mapping tables
├── data/
│   ├── 00_raw/          ← 🔒 ข้อมูลดิบ ห้ามแก้ไข (อยู่ใน Google Drive ของทีม ไม่ได้อยู่ใน git)
│   ├── 10_staging/      ← ข้อมูลที่ clean แล้ว (parquet)
│   ├── 20_intermediate/
│   ├── 30_warehouse/    ← warehouse.duckdb
│   └── 40_exports/      ← parquet / csv / sql สำหรับส่งงาน
├── src/rri/             ← โค้ด logic ทั้งหมดอยู่ที่นี่ (rri = Road Risk Intelligence)
├── notebooks/           ← เล่าเรื่องให้อาจารย์อ่าน (import จาก src/ เท่านั้น ไม่มี logic ในนี้)
├── tests/                ← unit test คู่กับ src/ ทุกโมดูล + tests/fixtures/ (ข้อมูลตัวอย่างขนาดเล็ก)
├── dashboard/            ← Streamlit app (Thai UI)
├── reports/              ← ไฟล์ html/png/csv สำหรับแนบรายงาน
└── scripts/              ← สคริปต์เสริม เช่น scripts/make_fixtures.py
```

## การติดตั้ง

โปรเจกต์นี้ใช้ [`uv`](https://docs.astral.sh/uv/) เป็นตัวจัดการ dependency หลัก

```bash
# ติดตั้ง uv (ถ้ายังไม่มี)
brew install uv          # macOS
# หรือ: curl -LsSf https://astral.sh/uv/install.sh | sh

# ติดตั้ง dependency ทั้งหมด + pre-commit hooks
make setup
```

**ถ้าไม่สามารถติดตั้ง `uv` ได้** ให้ใช้ venv + pip แทน:

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # หรือ pip install -e . แล้วตามด้วย pip install pytest ruff mypy ...
pre-commit install
```

### ข้อมูลดิบ (data/00_raw/)

ข้อมูลดิบ **ไม่ได้อยู่ใน git repo นี้** (ไฟล์ใหญ่เกินขีดจำกัดของ GitHub และเป็นนโยบายของทีม)
ต้องคัดลอกมาจาก Google Drive ของทีมด้วยตัวเองแล้ววางไว้ที่ `data/00_raw/` ให้มีไฟล์ครบ 9 ไฟล์ตามที่ระบุใน
`CLAUDE.md` §5 ก่อนจึงจะรัน pipeline เต็มรูปแบบได้

การทดสอบอัตโนมัติ (`make test`) **ไม่ต้องพึ่งข้อมูลจริง** — ใช้ไฟล์ตัวอย่างขนาดเล็กใน `tests/fixtures/`
ที่จำลอง edge case ต่าง ๆ ของข้อมูลจริงไว้แล้ว (ดูหัวข้อถัดไป)

## การรัน

```bash
make test        # รัน unit test ทั้งหมด (ใช้ tests/fixtures/ ไม่ใช้ข้อมูลจริง)
make lint         # ruff check + format check
make typecheck    # mypy --strict
make profile      # รัน notebook 01_data_profiling.ipynb ใหม่ทั้งหมด (ต้องมีข้อมูลจริงใน data/00_raw/)
```

สร้างไฟล์ fixtures ใหม่จากข้อมูลจริง (ถ้าข้อมูลดิบมีการเปลี่ยนแปลง):

```bash
uv run python scripts/make_fixtures.py
```

## แดชบอร์ด (Dashboard)

Streamlit dashboard 8 หน้า ตอบคำถามธุรกิจทั้ง 8 ข้อ (BQ1–BQ8) — UI เป็นภาษาอังกฤษ
(CLAUDE.md N10) อ่านข้อมูลจาก `data/30_warehouse/warehouse.duckdb` และเรียกโมเดลที่ persist
ไว้แล้วเท่านั้น ไม่มีการ recompute cleaning/feature/training ใน dashboard code

**Public URL:** https://car-insurance-accident-analysis-fst5y6vwgfrfd84gc6nowq.streamlit.app

รันในเครื่องตัวเอง:

```bash
uv run streamlit run dashboard/app.py
```

`data/30_warehouse/warehouse.duckdb`, `data/50_models/*.{joblib,pt}`, และ
`data/10_staging/rainfall.parquet` ถูก commit เข้า git โดยเจตนา (ข้อยกเว้นเฉพาะจุดใน
`.gitignore`) เพราะเป็น build artifact ที่ deploy แล้ว ไม่ใช่ข้อมูลดิบ — Streamlit Community
Cloud deploy ตรงจาก GitHub โดยไม่มี persistent storage ให้ mount จึงต้อง commit ไฟล์เหล่านี้
เพื่อให้แอปที่ deploy แล้วรันได้จริง

## มาตรฐานโค้ด

- ทุกฟังก์ชันต้องมี type hint ครบ, ผ่าน `mypy --strict`
- ทุกฟังก์ชัน public ต้องมี docstring (เหตุผลเชิงธุรกิจเขียนเป็นภาษาไทยได้ แต่ตัวโค้ดและชื่อตัวแปรเป็นภาษาอังกฤษเสมอ)
- Logic ทั้งหมดอยู่ใน `src/` เท่านั้น — notebook ทำหน้าที่แค่ import และเล่าเรื่อง
- ห้ามใช้ `print()` — ใช้ `structlog` เท่านั้น
- ห้าม hard-code path ใด ๆ — ทุก path ต้องผ่าน `config/settings.py`
- ทุกกฎการ clean ข้อมูลต้องมี unit test คู่กัน

## ทีม

นักศึกษาวิศวกรรมคอมพิวเตอร์ 4 คน มหาวิทยาลัยเกษตรศาสตร์
