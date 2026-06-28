# Python_fördjupning

# OMXS30 – Data Engineering Pipeline för tekniska indikatorer och klassificering

Detta projekt implementerar en modulär data engineering‑pipeline för att hämta, strukturera och bearbeta aktiedata från OMXS30‑index. Syftet är att skapa ett konsekvent och reproducerbart dataset baserat på tekniska indikatorer och en multiklass‑målvariabel (uppgång, nedgång, neutral rörelse). Pipelinens output används för modellering, EDA och feature importance‑analys.

main.py — Körlogik för data engineering‑pipeline
main.py initierar loggning, skapar mappstrukturen och kör samtliga builder‑moduler i rätt ordning. Filen fungerar som startpunkt för hela data engineering‑flödet och säkerställer att pipelineen är reproducerbar och modulärt uppbyggd.

## Översikt
Projektet består av fyra huvudsteg som körs automatiserat via `main.py`:

1. **PriceBuilder**  
   Hämtar och strukturerar historiska OHLCV‑värden för OMXS30‑bolagen via *yfinance*. Säkerställer datakvalitet, likviditet och konsekvent format.

2 **PriceCombinatoricsBuilder**

Genererar kombinatoriska features baserade på OHLCV‑data. Modulen skapar 2‑, 3‑, 4‑ och 5‑kombinationer av prisvariabler genom skillnader, kvoter och produkter. Syftet är att utöka feature‑rymden med matematiska relationer som kan fånga komplexa mönster i prisrörelser. Outputen används som ett komplement till tekniska indikatorer i den slutliga dataset‑sammanställningen.

3. **IndexBuilder**
   
Hämtar och strukturerar indexdata (t.ex. OMXS30) som används för att beräkna relativa prisrörelser och marknadskontext. Modulen skapar bland annat Index_Return och andra tidsserievariabler som används av TargetBuilder för att konstruera målvariabeln. IndexBuilder säkerställer att pipelineen har en stabil referenspunkt mot bredare marknadsrörelser.
   
5. **TechBuilderV3**  
   Beräknar tekniska indikatorer (t.ex. glidande medelvärden, RSI, volatilitet). Resultatet sparas i standardiserade feature‑filer.

6.**BandBuilder**
   
Skapar bandindelningar baserat på prisrörelser eller volatilitetsnivåer. Modulen grupperar observationer i diskreta band (t.ex. låg, medel, hög) genom percentiler eller fasta trösklar. BandBuilder används av TargetBuilder för att konstruera målvariabeln och säkerställer att klassindelningen är konsekvent över hela datasetet. Detta steg är centralt för att modellen ska kunna lära sig riktning och mönster i prisrörelser.

8. **TargetBuilderV3**  
   Konstruerar målvariabeln som en treklasstilldelning: uppgång, nedgång eller neutral rörelse. Detta minskar brus jämfört med exakta prisprognoser.

9. **DatasetBuilderV3**  
   Slår samman prisdata, indikatorer och målvariabel till ett komplett dataset för modellering.


10. **Modellingsfasen

def main():
    print("\n=== STEG 1: TRAIN ===")
    train_all_groups()

    print("\n=== STEG 2: OPTIMIZE ===")
    model_classes = [RFModel, HGBModel, CatBoostModel]
    optimize_all_groups(model_classes)

    print("\n=== STEG 3: RETRAIN MED BEST PARAM ===")
    retrained_models = retrain_group(model_classes)

    print("\n=== STEG 4: TEST ===")
    test_results = test_all_groups(retrained_models)

    print("\n=== STEG 5: EVALUATE ===")
    evaluation_results = evaluate_group(test_results)

    print("\n=== STEG 6: PLOTS ===")
    plot_all_groups(evaluation_results)

    print("\n=== KLART ===")
    

Steg 2: Modellering — Machine Learning Pipeline

model_main.py kör hela modellflödet: träning, optimering, retraining, testning, utvärdering och plotting. Filen använder tre modellklasser (Random Forest, Histogram Gradient Boosting och CatBoost) och genererar confusion matrix, classification report och feature importance för varje grupp.

Pipelineen består av följande steg:

Train – tränar grundmodeller på alla grupper

Optimize – kör hyperparameter‑optimering för varje modellklass

Retrain – tränar om modellerna med bästa parametrar

Test – testar modellerna på respektive grupp

Evaluate – sammanställer resultat och nyckeltal

Plot – genererar grafer för utvärdering
