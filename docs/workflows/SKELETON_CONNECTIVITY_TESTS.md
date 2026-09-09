# Schrittweise Tests für durchgängige 1-px-Skeletons

[Zur Dokumentationsübersicht](../README.md)

Diese Versuchsleiter lässt die aktuelle R0-Baseline unverändert. Jeder Kandidat
schreibt in einen eigenen Ordner unter `skeleton_connectivity_runs` oder in einen
eigenen nnU-Net-Trainerordner. R1 Full Resolution Only wird nicht verwendet.

## Gemeinsame Regeln

- Formales Panel: `one_px_skeleton_review/panel_manifest.csv` mit denselben 80 Zellen.
- Kalibrierung: zwölf andere Fold-0-Zellen in
  `skeleton_connectivity_runs/calibration_manifest.csv`.
- Auf der Kalibrierungsseite sind die Variantennamen sichtbar. Die formale Seite
  ist zellweise verblindet und balanciert A/B.
- Erst die CSV eines Experiments auswerten, dann mit dem nächsten Experiment
  beginnen.
- Die Somamaske wird in R2 und R3 nie verändert.

## 0. R0-Wahrscheinlichkeiten einmalig exportieren

```powershell
.\run_r0_probability_export.ps1
```

Der Befehl trainiert nicht. Er führt nur die Fold-0-Validierung des fertigen
R0-Checkpoints erneut aus und ergänzt die 283 `.npz`-Wahrscheinlichkeitsdateien.
Ist der Export vollständig, beendet sich das Skript ohne erneute Berechnung.

## 1. R2 – Hysterese

Kalibrierungsvarianten `T_low=0.30`, `0.20` und `0.10` erzeugen:

```powershell
.\run_experiment02_hysteresis.ps1
```

Danach öffnen:

`skeleton_connectivity_runs/R2_hysteresis/calibration_review/calibration.html`

Die Seite empfiehlt automatisch eine Einstellung nach der festgelegten Regel:
mehr als zwei markierte Fehlverbindungsfälle schließen eine Variante aus;
ansonsten gewinnt die häufigste beste Kontinuität, bei Gleichstand der höhere
Schwellenwert.

Beispiel für den formalen 80er-Test nach Auswahl von `0.20`:

```powershell
.\run_experiment02_hysteresis.ps1 -Stage Formal -TLow 0.20
```

Nach Abschluss der HTML-Bewertung die exportierte CSV prüfen:

```powershell
.\.venv\Scripts\python.exe .\evaluate_skeleton_ab_acceptance.py `
  --ratings "C:\Pfad\zur\ratings.csv" `
  --base-name "R0 aktuelle Baseline" `
  --candidate-name "R2 Hysteresis Tlow 0.2" `
  --automatic-summary .\skeleton_connectivity_runs\R2_hysteresis\tlow_0p200\summary.json
```

## 2. R3 – geometrische Rekonnektion

Nur starten, wenn nach R2 relevante Lücken bleiben. Standardmäßig wird R0 als
Basis verwendet. Hat R2 gewonnen, dessen Prediction-Ordner explizit übergeben:

```powershell
.\run_experiment03_reconnect.ps1 -Stage Calibration `
  -BasePredictions .\skeleton_connectivity_runs\R2_hysteresis\tlow_0p200 `
  -BaseName "R2 Hysteresis Tlow 0.2"
```

Die Seite vergleicht 6 px und 10 px. Bei gleicher Kontinuität ist 6 px die
konservative Wahl. R3 schreibt je Ausgangsmodell in einen eigenen Ordner, zum
Beispiel `R3_reconnect_from_R2-Hysteresis-Tlow-0.2`; so kann ein späterer Test
mit einer anderen Basis keine vorhandenen Kandidaten überschreiben. Für mehrere
Läufe mit demselben Anzeigenamen kann zusätzlich `-RunName` gesetzt werden.
Beispiel für den formalen Test:

```powershell
.\run_experiment03_reconnect.ps1 -Stage Formal -MaxGap 6 `
  -BasePredictions .\skeleton_connectivity_runs\R2_hysteresis\tlow_0p200 `
  -BaseName "R2 Hysteresis Tlow 0.2"
```

## 3. R4 – labelsichere Augmentation

Zuerst der isolierte 50-Epochen-Techniktest:

```powershell
.\run_experiment04_labelsafe_aug.ps1
```

Nur wenn Loss, Skeletonausgaben und Validierung technisch korrekt sind:

```powershell
.\run_experiment04_labelsafe_aug.ps1 -FullRun
```

Die Runner speichern TIFF und NPZ. Falls R2 oder R3 akzeptiert wurden, dieselbe
Rezeptur getrennt auf R0 und R4 anwenden. Beispiel:

```powershell
.\run_apply_skeleton_recipe.ps1 `
  -ModelValidation "C:\Pfad\zum\R4\fold_0\validation" `
  -OutputRoot .\skeleton_connectivity_runs\R4_selected_recipe `
  -TLow 0.20 -MaxGap 6
```

Anschließend den generischen formalen Vergleich erzeugen:

```powershell
.\run_skeleton_formal_review.ps1 `
  -BasePredictions "C:\Pfad\zu\R0_mit_gleicher_Rezeptur" `
  -BaseName "bisheriger Gewinner" `
  -CandidatePredictions "C:\Pfad\zu\R4_mit_gleicher_Rezeptur" `
  -CandidateName "R4 Label-safe Augmentation" `
  -OutputName "R4_formal_review"
```

## 4. R5 – DeepFlux-Hilfsziel

Nur beginnen, wenn R4 akzeptiert wurde, aber weiterhin sichtbare Lücken besitzt.

```powershell
.\run_experiment05_deepflux_aux.ps1
```

Nach bestandenem 50-Epochen-Techniktest:

```powershell
.\run_experiment05_deepflux_aux.ps1 -FullRun
```

R5 verwendet R4 als Basis und ergänzt nur im Training ein zweikanaliges
Radius-7-Richtungsfeld mit Gewicht 0.1. In Validierung und Inferenz liefert das
Netz weiterhin ausschließlich die normale semantische 0/1/2-Ausgabe.

## Abnahme

`evaluate_skeleton_ab_acceptance.py` setzt die geplanten Regeln um:

- Kontinuität: mindestens acht Netto-Siege;
- Fehlverbindungen, Soma-Anschluss, 1-px-Breite und Gesamturteil nicht schlechter;
- keine zusätzlichen leeren Skeletons;
- Soma-Dice-Abfall höchstens 0.005;
- höchstens 25 % mehr Skeletonpixel;
- Abfall des 1-px-Anteils höchstens 0.02.

Die automatische Tabelle enthält außerdem Skeleton-Präzision/Recall/Dice,
`beta0` der Vereinigung aus Skeleton und Soma, Soma-Anbindung und Endpunktzahl.
