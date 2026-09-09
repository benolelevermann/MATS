# Projektstruktur

[Zur Dokumentationsübersicht](README.md)

## Stabiler ausführbarer Kern

Die Webanwendungen und gemeinsam genutzten Module liegen in eigenen Ordnern:

| Ordner | Inhalt |
| --- | --- |
| `cell_pipeline_web/` | Modellinferenz, Hysterese, Zell-Review und Export |
| `blind_selection_web/` | manuelle Auswahl vor der automatischen Segmentierung |
| `instance_annotation_web/` | Erstellen und Prüfen von Instanz-IDs |
| `custom_trainers/` | clDice- und Skeleton-Recall-Trainer für nnU-Net |
| `r_pipeline/` | Fiji/SNT-Export und scEvoView-Auswertung |
| `tests/` | Unit- und Integrationstests |
| `tools/` | eigenständige Hilfswerkzeuge |

## Skripte im Projektstamm

Die operativen Skripte bleiben aus Kompatibilitätsgründen im Projektstamm.
Mehrere PowerShell-Workflows leiten aus `$PSScriptRoot` die Pfade zu Python,
nnU-Net, Daten und Checkpoints ab. Die Dateinamen bilden dennoch klar getrennte
Gruppen:

| Präfix beziehungsweise Name | Aufgabe |
| --- | --- |
| `run_*.ps1` | direkt ausführbare Workflows und Trainingsläufe |
| `build_*.py`, `create_*.py` | Trainingsdatensätze und Zwischenprodukte erzeugen |
| `prepare_*.py`, `select_*.py`, `rasterize_*.py` | Datenimport und Trainingsvorbereitung |
| `apply_*.py`, `postprocess_*.py`, `simple_postprocess_*.py` | Nachverarbeitung und Zelltrennung |
| `make_*.py` | HTML-Berichte, Galerien und Übersichten |
| `compare_*.py`, `evaluate_*.py`, `sweep_*.py` | Modellvergleich und Parameterauswertung |
| `train_*.py`, `continue_*.py` | Training und Fortsetzung von Checkpoints |
| `scEvoView_pipeline.Rmd` | zentrale interaktive R-Auswertung |

Ein späteres Verschieben dieser Skripte sollte als eigener Refactor erfolgen,
bei dem sämtliche PowerShell-Pfade und Python-Imports gemeinsam angepasst und
vollständig getestet werden.

## Nicht versionierte Arbeitsdaten

Die großen lokalen Ordner sind absichtlich nicht Teil des Repositories. Dazu
gehören insbesondere `nnUNet_raw/`, `nnUNet_preprocessed/`, `nnUNet_results/`,
datierte Experimentordner, Web-Pipeline-Läufe und Evo-Zellexporte. Details
stehen in [DATA_STORAGE.md](DATA_STORAGE.md).
