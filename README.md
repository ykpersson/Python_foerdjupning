# Python_fördjupning

# OMXS30 – Data Engineering Pipeline för tekniska indikatorer och klassificering

Detta projekt implementerar en modulär data engineering‑pipeline för att hämta, strukturera och bearbeta aktiedata från OMXS30‑index. Syftet är att skapa ett konsekvent och reproducerbart dataset baserat på tekniska indikatorer och en multiklass‑målvariabel (uppgång, nedgång, neutral rörelse). Pipelinens output används för modellering, EDA och feature importance‑analys.

## Översikt
Projektet består av fyra huvudsteg som körs automatiserat via `main.py`:

1. **PriceBuilder**  
   Hämtar och strukturerar historiska OHLCV‑värden för OMXS30‑bolagen via *yfinance*. Säkerställer datakvalitet, likviditet och konsekvent format.

2. **TechBuilderV3**  
   Beräknar tekniska indikatorer (t.ex. glidande medelvärden, RSI, volatilitet). Resultatet sparas i standardiserade feature‑filer.

3. **TargetBuilderV3**  
   Konstruerar målvariabeln som en treklasstilldelning: uppgång, nedgång eller neutral rörelse. Detta minskar brus jämfört med exakta prisprognoser.

4. **DatasetBuilderV3**  
   Slår samman prisdata, indikatorer och målvariabel till ett komplett dataset för modellering.

Därefter körs:

- **EDA + grouping** (volatilitet, beteendemönster, klustring)  
- **Feature importance** med Random Forest, Histogram Gradient Boosting och CatBoost

## Pipeline‑struktur

main.py
├── PriceBuilder
├── TechBuilderV3
├── TargetBuilderV3
├── DatasetBuilderV3
├── EDA (run_full_eda_pipeline)
└── Feature importance (compute_rf_feature_importance)
├── ticker_exp.py
├── support_functions.py
├── logger.py
├── setting.py
└── /data (skapas automatiskt)

Output genereras i data/ och inkluderar:

rådata

tekniska indikatorer

målvariabel

slutligt dataset

EDA‑sammanfattningar

feature importance‑resultat








