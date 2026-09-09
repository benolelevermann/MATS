# MATS

MATS ist die lokale Pipeline für die Segmentierung mikroskopischer Zellbilder,
die Hysterese-Nachverarbeitung 1-px-breiter Skeletons und die konservative
Extraktion einzelner Zellen. Dieses Repository versioniert den reproduzierbaren
Projektteil: Quellcode, Startskripte, Tests, Konfiguration und Dokumentation.

Rohbilder, Trainingsdaten, Modell-Checkpoints und erzeugte Resultate bleiben
wegen ihrer Größe auf dem lokalen Laufwerk beziehungsweise dem Datenserver.

## Aktueller Standard

- Modell: Dataset 141, finaler Checkpoint nach 1.000 Epochen
- Trainer: `nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug`
- Plans: `nnUNetPlans139Exact`, 2D, Patchgröße 160 x 160 px
- Klassen: Hintergrund 0, Skeleton 1, Soma 2
- Hysterese: `T_high=0.647`, `T_low=0.20`

Der lokale Modellcheckpoint und sämtliche Bilddaten sind absichtlich nicht in
Git enthalten. Der erwartete Checkpoint liegt unter:

```text
nnUNet_results/Dataset141_dataset139_plus_net139_reviewed_cells/
  nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug__nnUNetPlans139Exact__2d/
  fold_0/checkpoint_final.pth
```

## Schnellstart

Nach dem Klonen:

```powershell
git clone --recurse-submodules https://github.com/benolelevermann/MATS.git
cd MATS
```

Die Python-Umgebung und die in `requirements-lock.txt` dokumentierten Pakete
müssen lokal eingerichtet werden. Anschließend sind die wichtigsten
Einstiegspunkte:

```powershell
# Neue Trainingszellen erzeugen und manuell prüfen
.\run_net141_training_data_pipeline.ps1

# Zellen für die Evo/R-Pipeline erzeugen
.\run_net141_evo_pipeline.ps1

# Instanz-IDs in den MATS-Übersichtsbildern annotieren
.\run_instance_annotation_tool.ps1

# Mehrzellbilder aus Dataset141 synthetisieren und für Fine-Tuning vorbereiten
.\run_build_dataset143_synthetic_multicell.ps1 -Preprocess

# Tests ohne zusätzliche Test-Abhängigkeit
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Weitere Details stehen in:

- [Dokumentationsübersicht](docs/README.md)
- [Projektstruktur](docs/PROJECT_STRUCTURE.md)
- [Daten- und Modellspeicherung](docs/DATA_STORAGE.md)
- [Web-Pipeline](docs/workflows/CELL_PIPELINE_WEB_README.md)
- [Instanzannotationswerkzeug](docs/workflows/INSTANCE_ANNOTATION_TOOL_README.md)
- [Dataset143](docs/workflows/DATASET143_SYNTHETIC_MULTICELL_README.md)
- [Skeleton-Recall-Workflow](docs/workflows/SKELETON_RECALL_TO_EVO_WORKFLOW.md)
- [Skeleton-Recall-Trainer](custom_trainers/skeleton_recall/README.md)

## Projektstruktur

```text
MATS/
├── blind_selection_web/       manuelle Zellvorauswahl
├── cell_pipeline_web/         Inferenz-, Review- und Export-Webanwendung
├── custom_trainers/           externe nnU-Net-Trainer
├── docs/
│   ├── workflows/             Bedienung und reproduzierbare Abläufe
│   ├── research/              Literatur und Methodenentscheidungen
│   └── archive/               historische Projektstände
├── instance_annotation_web/   Werkzeug für Instanz-IDs
├── r_pipeline/                scEvoView-/R-Auswertung und Fiji-Export
├── tests/                     automatische Tests
├── tools/                     externe Hilfswerkzeuge
├── run_*.ps1                  direkt ausführbare Workflows
└── *.py                       Pipeline- und Analyseskripte
```

Die ausführbaren `run_*.ps1`-Dateien bleiben bewusst im Projektstamm: Sie
verwenden ihren eigenen Speicherort als stabile Projektwurzel. Die Präfixe
`build_`, `prepare_`, `make_`, `compare_`, `evaluate_` und `run_` gliedern die
operativen Skripte nach Aufgabe. Eine vollständige Zuordnung steht in der
[Projektstruktur](docs/PROJECT_STRUCTURE.md).

## Änderungen zu GitHub übertragen

Für spätere Updates genügt:

```powershell
.\update_github.ps1 -Message "Kurze Beschreibung der Änderung"
```

Das Skript zeigt vor dem Commit alle vorgesehenen Dateien, blockiert neue
Dateien über 50 MiB und fragt vor dem Commit noch einmal nach. Mit `-NoPush`
wird nur lokal committed.

## Lokale Einrichtung

Das Repository enthält Code, Tests und Dokumentation, aber keine Rohbilder,
Trainingsdaten, Checkpoints, Vorhersagen oder Review-Ausgaben. Die aktuell
verwendeten Python-Pakete sind in `requirements-lock.txt` dokumentiert. Das
CUDA-PyTorch-Paket muss gegebenenfalls über den zur CUDA-Version passenden
PyTorch-Paketindex installiert werden.

NeuroTreeTracer wird als Git-Submodul eingebunden:

```powershell
git clone --recurse-submodules <repository-url>
git submodule update --init --recursive
```

Die lokal benötigten MATLAB/Octave-Kompatibilitätsänderungen sind separat in
`patches/neurotreetracer-local.patch` versioniert. Nach einem frischen Clone:

```powershell
git -C external/NeuroTreeTracer apply ../../patches/neurotreetracer-local.patch
```

Zusätzliche lokale Programme für den vollständigen Evo-Export:

- Fiji/ImageJ mit SNT
- GNU Octave für NeuroTreeTracer
- CUDA-fähige PyTorch-Installation für GPU-Inferenz und Training

## Datenhaltung

Die `.gitignore` schließt insbesondere folgende lokale Verzeichnisse aus:

- `nnUNet_raw/`, `nnUNet_preprocessed/`, `nnUNet_results/`
- `web_pipeline_runs*/`, `review_dataset/`
- `instance_annotation_dataset142/`
- alle datierten Experiment- und Vergleichsordner

Diese Daten müssen getrennt gesichert werden. Git ersetzt kein Backup der
Mikroskopiebilder oder trainierten Modelle. Einzelheiten und die erwarteten
lokalen Pfade stehen in [Daten- und Modellspeicherung](docs/DATA_STORAGE.md).
