# Projekt-Handoff: Glioblastom-Zellsegmentierung (Stand 2026-08-20)

[Zur Dokumentationsübersicht](../README.md)

Dieses Dokument fasst den Projektstand zusammen, damit ein LLM in einem neuen Chat
ohne Vorwissen weiterarbeiten kann.

---

## 1. Umgebung

| | |
|---|---|
| Projektwurzel | `C:\Ole\20260721_CellClassification_v2` |
| Betriebssystem | Windows 11 Pro, PowerShell 5.1 (Git Bash zusätzlich verfügbar) |
| Python | `.venv\Scripts\python.exe` (3.14), numpy 2.4.4, scipy 1.18, tifffile 2026.7.14 |
| Deep Learning | nnU-Net v2, torch 2.13.0+cu130 |
| GPU | NVIDIA RTX 5060 Ti, 17,1 GB VRAM |
| RAM | 61,6 GB |
| MATLAB | R2022b installiert, **Lizenz abgelaufen** (License Manager Error -10) |
| Octave | 11.3.0 unter `C:\Users\01.144_1\AppData\Local\Programs\GNU Octave\Octave-11.3.0` |
| Fiji | `C:\Program Files\Fiji.app\ImageJ-win64.exe` |
| Kein Git-Repo | Die Projektwurzel steht nicht unter Versionskontrolle. |

nnU-Net-Umgebungsvariablen müssen vor jedem Aufruf gesetzt werden:

```powershell
$env:nnUNet_raw          = "C:\Ole\20260721_CellClassification_v2\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\Ole\20260721_CellClassification_v2\nnUNet_preprocessed"
$env:nnUNet_results      = "C:\Ole\20260721_CellClassification_v2\nnUNet_results"
```

---

## 2. Worum es geht

Segmentierung von Glioblastomzellen in Fluoreszenzaufnahmen. Ein nnU-Net segmentiert
semantisch in drei Klassen:

```
0 = Hintergrund, 1 = Skelett (Neuriten), 2 = Soma
```

Danach folgt eine Nachbearbeitungskette, die Skelettfragmente verbindet, Neuriten
einzelnen Somas zuordnet und pro Zelle einen Crop exportiert. Diese Crops gehen in die
"Evo"-Pipeline (R-basierte Merkmalsextraktion) weiter.

Wichtig: Klasse 1 ist ein **1 Pixel breites Skelett**, kein dicker Neuritenkörper.
Das ist für nachgelagerte Verfahren relevant, die Distanztransformationen benutzen.

---

## 3. Datensätze

`nnUNet_raw` enthält Dataset123 bis Dataset138. Relevant sind:

| Dataset | Fälle | Beschreibung |
|---|---|---|
| Dataset136 | 2013 | Basis, alle Zellen |
| Dataset137 | 2413 | 136 + 400 manuell getracte DIV10-Zellen (nonBleb, nonDMSO) |
| **Dataset138** | **1412** | **NEU: manuell QC-gefilterte und enger zugeschnittene Teilmenge von 137** |

### Dataset138 im Detail

Voller Name: `Dataset138_cleanSingleCell_soma_skeleton_recrop`

Entstanden am 2026-08-19 aus Dataset137 mittels
`build_dataset138_clean_single_cell_from_dataset137.py build`, gesteuert durch das
manuell durchgesehene Manifest
`dataset138_clean_single_cell_candidates/selection_manifest_single_cell_qc.csv`.

- **1412 von 2413** Fällen ausgewählt (`selected_for_dataset == 1`), 1001 ausgeschlossen
- Die manuelle Durchsicht hat den Auto-Vorschlag deutlich korrigiert:
  **353 Zellen manuell aufgenommen**, die der Detektor auf `review`/`rejected` hatte,
  und **461 manuell verworfen**, die er auf `accepted` hatte
- Jedes Paar ist auf die Bounding-Box von Skelett+Soma plus 8 px Rand zugeschnitten.
  Kein Rescaling, keine Intensitätsänderung — jedes Ausgabepixel ist byte-identisch
  zum Quellpixel. Verifiziert für alle 1412 Paare.
- Median-Bildgröße: 122×123 px (137) → **90×89 px** (138); Median 61 % der Fläche behalten
- Case-IDs unverändert, jede Zelle ist 1:1 zu 137 rückverfolgbar
- **Splits: frisch von nnU-Net gewürfelt** (bewusste Entscheidung). Die Train/Val-Zugehörigkeit
  ist damit *nicht* identisch zu 137.

---

## 4. Training Dataset138

```powershell
python -m nnunetv2.run.run_training 138 2d 0 -tr nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x -p nnUNetPlans
```

Der Custom-Trainer `nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x` liegt in
`custom_trainers/skeleton_recall/` und ist ins venv installiert.

1000 Epochen, fold 0, abgeschlossen 2026-08-19.

| | Dataset137 | Dataset138 |
|---|---|---|
| Pseudo-Dice Skelett | 0,6004 | 0,6662 |
| Pseudo-Dice Soma | 0,8732 | 0,9357 |
| bestes EMA | 0,7391 | 0,8052 |
| Patch-Size | 128×128 | 96×96 |
| Batch-Size | 111 | 62 |

### ACHTUNG — diese Zahlen sind NICHT vergleichbar

Der scheinbare Zugewinn ist zu einem unbekannten Teil ein Artefakt:

1. Dataset138 hat einen frisch gewürfelten Validierungssatz
2. Dieser besteht ausschließlich aus manuell als sauber markierten Zellen, ist also
   per Konstruktion leichter
3. Die engeren Crops erhöhen den Vordergrundanteil, was Dice mechanisch anhebt

Ein Teil des Zugewinns misst also, dass die Aufgabe leichter geworden ist, nicht dass
das Netz besser generalisiert. Für einen sauberen Vergleich müsste 138 mit den
gefilterten 137-Splits neu trainiert werden (`--source-splits` beim Bauen).

---

## 5. Predictions mit Dataset138

Aufrufmuster:

```powershell
nnUNetv2_predict -i <input_dir> -o <output_dir> -d 138 -c 2d -f 0 `
  -tr nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x -p nnUNetPlans --save_probabilities
```

Eingaben müssen `<case_id>_0000.tif` heißen und **2D** sein.

Durchsatz gemessen: ca. **8,3 Minuten pro 90 Mpx** auf der RTX 5060 Ti.

### 5.1 NewImage2 (`20280812_newTest2/08_predictions_dataset138`)

Bild 5040×5056. Vergleich gegen die vorhandene 137-Prediction desselben Bildes:

| | 137 | 138 |
|---|---|---|
| Skelettpixel | 31 245 | 35 497 (+13,6 %) |
| Somapixel | 420 554 | 499 049 (+18,7 %) |
| Soma-Instanzen ≥200 px | 153 | 190 (+24 %) |
| größte Skelettkomponente | 372 | 511 |

IoU zwischen den Netzen: Skelett 0,550, Soma 0,790.

### 5.2 Objektivtest (`20260817_ObjectiveTest`)

16 Aufnahmen desselben Präparats unter verschiedenen Objektiven, Belichtungen und
Binning-Einstellungen, mit **beiden** Netzen auf **identischen** Eingaben gerechnet:

- Eingaben: `02_dataset137_prediction_only_10X/00_flat_inference_inputs/`
- 137-Predictions: `02_dataset137_prediction_only_10X/01_dataset137_predictions/`
- 138-Predictions: `04_dataset138_flat/01_predictions/`
- **Auswertung: `05_overview_137_vs_138/`** (HTML-Galerie, CSV, JSON),
  erzeugt mit `make_objectivetest_overview.py`

Kernbefunde:

- Dataset138 findet auf **allen 16** Bildern mehr Skelett (+11 % bis +71 %)
- Beim Anteil "Skelett am Soma" liegt 138 bei **14 von 16** vorn, Median +3,0 Prozentpunkte
- Zwei Ausnahmen: `10X_CF` (26,3 % → 24,0 %) und vor allem **`10X_WF` (19,3 % → 6,1 %)**
- IoU Skelett durchgehend nur 0,37–0,59 — die Netze sind überall deutlich verschieden
- **Binning ist der schärfste Trenner:** `binningon` liefert 24–54 Soma-Instanzen,
  `nobinning` bei sonst gleicher Optik 149–462. Beste Werte: `50xLP_nobinning` mit
  langer Belichtung.

**40X_CF und 40X_WF wurden bewusst ausgelassen** — 35406×35369 = 1252 Mpx, ca. 2 Stunden
pro Bild und Netz. Für 137 existieren sie ebenfalls nicht.

### 5.3 Rollingball-Hintergrundabzug — uneinheitlich

Zwei Vergleiche mit widersprüchlichem Ergebnis:

| Bild | Skelett am Soma ohne RB | mit RB |
|---|---|---|
| `10X_CF` | 18,4 % | **22,6 %** (besser) |
| `CF_50xLP_10x_1_Exp500ms_nobinning` | 29,4 % | **24,8 %** (schlechter) |

Beim zweiten Bild: +18,8 % Skelett, aber −13,9 % Soma-Instanzen, und die größte
Skelettkomponente verdoppelt sich (1002 → 2078).

**Interpretationsfalle:** "Skelett am Soma" ist ein Bruch. Rollingball verändert Zähler
*und* Nenner. Die Metrik taugt zum Vergleich von Aufnahmen bei **gleicher** Vorverarbeitung,
aber nicht für mit-gegen-ohne-Rollingball. Ohne annotierte Referenzregion lässt sich nicht
entscheiden, ob die zusätzlichen Skelettpixel echt sind.

---

## 6. Die Evo-Pipeline (Nachbearbeitung → Zell-Crops)

Ein einziges Skript orchestriert die ganze Kette:

```powershell
run_skeleton_recall_to_evo_workflow.ps1 `
  -RunRoot  <Laufordner> `
  -InputDir <Laufordner>\01_inputimages `
  -DatasetId 138 `
  -Trainer nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x
```

Optional `-RunFiji` für die SNT-Finalisierung (erzeugt `bounds.zip`, `locations.zip`
pro Fall und validiert die Zellordner).

Stufen und Ausgabeordner unterhalb von `-RunRoot`:

| Stufe | Ordner | Skript |
|---|---|---|
| 1 Inferenz | `02_predictions_skeleton_recall` | nnUNetv2_predict |
| 2 Soma-Vorbereitung | `03_soma_preparation` | `prepare_somas_for_assignment.py` |
| 3 Lückenschluss | `04_conservative_no_loss_postprocessing` | `postprocess_net129_no_loss.py` |
| 4 Zuordnung | `05_neurocytoII_inspired_assignment` | `compare_cell_assignment_methods.py` |
| 5 Repräsentantenwahl | `06_overlap_representative_selection` | `select_overlap_representative_cells.py` |
| 6 **Crops für Evo** | `07_evo_cells_matrix_forest_inspired` | `export_assigned_cells_for_evo.py` |
| 7 Galerie | `08_evo_crop_overviews_matrix_forest_inspired` | `make_cells_overview.py` |

Das Skript ist idempotent: fertige Stufen werden übersprungen, ein erneuter Aufruf mit
`-RunFiji` zieht nur den Fiji-Schritt nach.

Zuordnungsmethode per Default `matrix_forest_inspired`. Stufe 4 rechnet immer alle drei
Methoden (`topology_baseline`, `matrix_forest_inspired`, `gcut_inspired`) für die QC-Seite
`comparison.html`.

Format eines Evo-Zellordners (`cell0001/`):
`raw.tif`, `skeleton.tif`, `soma.tif`, `seg.tif`, `cell_mask.tif`, `seg-000.swc`,
`seg.csv`, `seg.traces`, `bounds.json/.zip`, `location.json/.zip`, `metadata.json`

Crop-Parameter (fest im Workflow): Margin 48, Mindestgröße 128, Edge-Clearance 3, quadratisch.

### Laufzeiten (gemessen)

Vorlauf `20260812_my_run`, 2 Bilder à ~9 Mpx: **37,9 Minuten gesamt**. Die Verteilung ist
extrem ungleich — zwei Stufen fressen 90 % der Zeit:

- Lückenschluss: 11,4 min (Bleb) bzw. 8,5 min (DMSO)
- Repräsentantenwahl: 13,3 min
- alles andere: Sekunden bis 2 Minuten

**Der Lückenschluss skaliert mit der Fragmentzahl, nicht mit der Bildfläche.** Eine
Hochrechnung über die Pixelzahl unterschätzt ihn deutlich.

---

## 7. TrogoCellStateTest — ABGESCHLOSSEN

Ordner: `20260817_TrogoCellStateTest`, gelaufen am 2026-08-20, Dauer **1 h 50 min**.

Zwei Bilder in `01_inputimages`: `PBS_0000.tif` und `Syn_0000.tif`, je 4744x4768,
**dtype uint8** (Training war uint16).

```powershell
run_skeleton_recall_to_evo_workflow.ps1 `
  -RunRoot "C:\Ole60721_CellClassification_v260817_TrogoCellStateTest" `
  -InputDir "C:\Ole60721_CellClassification_v260817_TrogoCellStateTest_inputimages" `
  -DatasetId 138 -Trainer nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x -RunFiji
```

### Ergebnis

| | PBS | Syn |
|---|---|---|
| Skelettpixel (verlustfrei) | 72 754 | 130 354 |
| Soma-Instanzen behalten | 382 | 343 |
| davon zu klein verworfen | 74 | 207 |
| sichere Zellen (matrix_forest_inspired) | 223 | - |
| **exportierte Zell-Crops** | **242** | **228** |
| Fiji-finalisiert | ja | ja |
| Validator | 242/242 gueltig | 228/228 gueltig |

Zuordnungsvergleich PBS: topology_baseline 187 / matrix_forest_inspired 223 / gcut_inspired 259.

Das Dateiset jedes Zellordners ist **identisch zum Referenzlauf** `20260812_my_run`.
Die Crops sind bereit fuer die Evo-Pipeline.

Zum Vergleich: `20260812_my_run` lieferte 73 bzw. 72 Zellen bei 2,5-fach kleinerer Bildflaeche.

### uint8 war unkritisch

Die Sorge wegen der geringen Bittiefe (Median 8, p99,9 = 54 bei PBS, also effektiv ~6 Bit
statt 16) hat sich nicht bestaetigt - 382 bzw. 343 detektierte Somas und verlustfreies
Skelett. Auffaellig bleibt, dass bei Syn **207 Somas als zu klein verworfen** wurden
gegenueber 74 bei PBS, bei fast doppelt so viel Skelett. Ob das biologische Differenz
zwischen den Bedingungen ist oder zerfallende Soma-Detektionen, ist ungeklaert - die
Galerie `08_evo_crop_overviews_matrix_forest_inspired/Syn/all_cells_overview.png` ist
der schnellste Pruefweg.

## 8. NeuroTreeTracer (Nebenprojekt, funktionsfähig)

Getestet wurde der Tracing-Algorithmus aus Kayasandik et al., *Automated sorting of neuronal
trees in fluorescent images of neuronal networks using NeuroTreeTracer*, Sci Rep 8:6450 (2018),
doi:10.1038/s41598-018-24753-w.

Von den vier Stufen des Papers (Denoising, SVM-Segmentierung, Soma-Detektion, Tree Extraction)
ist **nur die vierte** relevant — die ersten drei ersetzt das nnU-Net. Das Paper sagt dazu:
"We assume that the binary segmented image and the soma masks are given as input."

Repo geklont nach `external/NeuroTreeTracer`
(github.com/cihanbilge/AutomatedTreeStructureExtraction).

Einstiegspunkt: `runCenterLineParallel(inputSeg, inputSoma, option)`

| Eingabe | Typ | Bedeutung |
|---|---|---|
| `inputSeg` | uint8 0/1 | **binäre** Segmentierung inkl. Soma |
| `inputSoma` | uint8 0/1 | **binäre** Soma-Maske; Instanzen trennt der Code selbst |

Eigene Brückenskripte:

- `neurotreetracer_export_crop.py` — Prediction → `inputSeg.mat`/`inputSoma.mat`
- `run_neurotreetracer.m` — Batch-Treiber (setzt `option.manual=0`, speichert `-v7`)
- `run_neurotreetracer_octave.ps1` — Octave-Wrapper
- `neurotreetracer_import_traces.py` — Traces → TIFF/CSV/Overlay
- `octave_shims/bwconvhull.m` — fehlt im Octave-image-Paket
- `20260819_NeuroTreeTracerTest/octave_portability.patch` — 3 Dateien, rücknehmbar mit `git apply -R`

Ergebnis auf einem Bleb-Crop (384×384, 5 Somas): 9 Neuriten über 4 Somas, 975 Trace-Punkte,
257 s. Verifiziert: 100 % der Punkte auf `inputSeg`, alle Traces starten mit Distanz 0 am Soma.

**Offen:** Der Kontrolllauf auf den Originaldaten des Papers wurde nach 48 Minuten
abgebrochen. Ob die Octave-Portierung die publizierten Ergebnisse reproduziert, ist
**nicht belegt**. Eingaben liegen bereit in
`20260819_NeuroTreeTracerTest/90_control_paper_reference/`.

---

## 9. Fallstricke, die Zeit gekostet haben

Diese Punkte sind nicht offensichtlich und haben jeweils zu Fehlern geführt:

1. **`20X_LSM_CF_0000.tif` in `20260817_ObjectiveTest/01_inputimages` ist ein z-Stack**
   (41, 4152, 4152), kein 2D-Bild. nnU-Net bricht daran ab. Beim Inventarisieren von
   TIFFs immer `series[0].shape` prüfen, nicht `pages[0]`. Die aufgelöste 2D-Fassung liegt
   in `00_flat_inference_inputs/`.

2. **`connComp.m` sortiert Komponenten nach Größe absteigend.** Die `soma_id` in
   NeuroTreeTracer-Traces ist ein **Größenrang**, kein Rasterindex. `scipy.ndimage.label`
   nummeriert in Rasterreihenfolge — wer beides gleichsetzt, ordnet Traces den falschen
   Zellen zu.

3. **MATLAB-Linearindizes sind 1-basiert und spaltenweise (column-major).**
   Zeile = `(k-1) % H`, Spalte = `(k-1) // H`. Falsch herum transponiert das Ergebnis
   unbemerkt.

4. **`createRectangles` in NeuroTreeTracer alloziert 18 × 10 × 360 = 64 800 bildgroße
   Logicals.** Speicher = 64 800 · H · W Bytes, gemessen exakt. 512×512 kostet 17 GB,
   ein volles 2964×3072-Bild wären 590 GB. Crops sind Pflicht.

5. **Octave fehlt `bwconvhull`**, und `strel('disk',R)` verlangt dort `N=0` (MATLAB nutzt
   per Default N=4, eine *approximierte* Scheibe). `bwboundaries` verlangt in Octave die
   numerische Konnektivität als zweites Argument.

6. **`genseed.m` enthält ein unbedingtes `figure; imshow(...)`**, das *nicht* hinter
   `option.manual` steht — der Code öffnet also auch im vermeintlich vollautomatischen
   Modus Fenster. Bug im Original.

7. **`Script.m` von NeuroTreeTracer wird mit `option.manual=1` ausgeliefert** und blockiert
   damit in nicht-interaktiven Sitzungen endlos auf `input()`.

8. **`scipy.io.loadmat` kann MATLABs `-v7.3` (HDF5) nicht lesen.** Immer `-v7` speichern.

9. **Der `-RunFiji`-Schritt bricht mit Exit-Code 1 ab, obwohl er erfolgreich ist.**
   `ImageJ-win64.exe` ist ein Launcher, der sich unter Windows neu startet und sofort
   zurueckkehrt. Das Workflow-Skript prueft daraufhin auf `bounds.zip`/`locations.zip`,
   waehrend Fiji noch rechnet, und wirft. Fiji laeuft im Hintergrund weiter und wird
   fertig. **Nie dem Exit-Code trauen, immer auf der Platte pruefen:** `bounds.zip`,
   `locations.zip`, `_fiji_finalize_log.txt` auf Fallebene und `seg.traces` in jedem
   Zellordner. Das Log endet bei Erfolg mit `GLOBAL OK bounds.zip and locations.zip (N ROIs each)`.
   Ein Fall pro Aufruf; fuer den zweiten Fall erneut aufrufen (das Skript erkennt den
   fertigen Fall am Log und ueberspringt ihn). Abschliessend validieren mit
   `r_pipeline/validate_r_pipeline_cell_folders.py --input-dir <caseCellRoot>`.

10. **Für den Lückenschluss und NeuroTreeTracer nicht die rohe Prediction verwenden**,
   sondern `04_conservative_no_loss_postprocessing/<case>/15_completed_skeleton_and_soma_0-1-2.tif`.
   Beispiel Bleb: Anteil Skelett am Soma steigt von 16,7 % auf 57,4 %, größte Komponente
   von 313 auf 697 px.

---

## 10. Wichtige eigene Skripte in der Projektwurzel

| Datei | Zweck |
|---|---|
| `build_dataset138_clean_single_cell_from_dataset137.py` | `prepare` (QC-Galerie + Manifest) und `build` (Dataset bauen) |
| `run_skeleton_recall_to_evo_workflow.ps1` | Komplette Kette Inferenz → Evo-Crops |
| `run_paper_guided_skeleton_to_evo.ps1` | Alternative, paper-geführte Kette |
| `prepare_somas_for_assignment.py` | Soma-Löcher füllen, Kleinstsomas filtern |
| `postprocess_net129_no_loss.py` | Verlustfreier Lückenschluss im Skelett |
| `compare_cell_assignment_methods.py` | Drei Zuordnungsmethoden + QC-HTML |
| `select_overlap_representative_cells.py` | Repräsentanten bei Überlappung |
| `export_assigned_cells_for_evo.py` | Per-Zell-Crops im Evo-Format |
| `make_cells_overview.py` | Crop-Galerie |
| `make_objectivetest_overview.py` | 137-gegen-138-Auswertung + HTML-Galerie |
| `neurotreetracer_export_crop.py` / `_import_traces.py` | NeuroTreeTracer-Brücke |
| `summarize_nnunet_training_runs.py` | Trainingsläufe zusammenfassen |

---

## 11. Offene Punkte

1. **Der 137-gegen-138-Vergleich ist nicht sauber quantifizierbar**, solange keine
   annotierte Referenzregion existiert. Alle bisherigen Aussagen beruhen auf Proxy-Metriken
   (Skelettmasse, Soma-Instanzen, Anteil Skelett am Soma). Für eine belastbare Aussage:
   entweder ein Bild mit manueller Annotation, oder 138 mit den gefilterten 137-Splits neu
   trainieren, damit die Validierungssätze deckungsgleich sind.
2. **NeuroTreeTracer-Kontrolllauf** auf den Paper-Originaldaten steht aus.
3. **40X-Bilder** (1252 Mpx) sind mit keinem Netz gerechnet.
4. **Rollingball-Effekt ist ungeklärt** — hilft bei einem Bild, schadet beim anderen.

---

## 12. Wie man mit mir weiterarbeitet

Nützliche Konventionen in diesem Projekt:

- Läufe liegen in datierten Ordnern (`YYYYMMDD_Name`) mit durchnummerierten Stufen
  (`01_inputimages`, `02_...`).
- Fast jede Stufe schreibt `run_summary.json` und `_script_version.txt` — daran lässt
  sich nachvollziehen, welches Skript in welcher Version einen Ordner erzeugt hat.
- Skripte in der Projektwurzel, Ausgaben in den Laufordnern. Quelldatensätze werden nie
  verändert.
- Lange Läufe im Hintergrund starten und den Fortschritt über Ordnerstand und
  Konsolenausgabe verfolgen.
