# Daten- und Modellspeicherung

[Zur Dokumentationsübersicht](README.md)

## Was in Git gehört

Git versioniert in diesem Projekt ausschließlich reproduzierbare, kleine
Artefakte:

- Python-, PowerShell-, R-, JavaScript-, HTML- und CSS-Quellcode
- Tests und Konfigurationsdateien
- Dokumentation und Literaturzusammenfassungen
- kleine Patches und das NeuroTreeTracer-Submodul

## Was außerhalb von Git bleibt

Folgende Inhalte sind durch `.gitignore` ausgeschlossen und müssen separat auf
dem Datenserver gesichert werden:

- `.tif`, `.tiff`, `.npy`, `.npz` und andere Bild- oder Arraydaten
- `.pth`, `.pt` und `.ckpt` Modell-Checkpoints
- `nnUNet_raw/`, `nnUNet_preprocessed/` und `nnUNet_results/`
- Web-Pipeline-Läufe, Review-Ordner und exportierte Evo-Zellen
- datierte Trainings-, Vorhersage- und Vergleichsordner

Der aktuelle Dataset141-Checkpoint wird lokal an folgendem relativen Pfad
erwartet:

```text
nnUNet_results/Dataset141_dataset139_plus_net139_reviewed_cells/
  nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1xLabelSafeAug__nnUNetPlans139Exact__2d/
  fold_0/checkpoint_final.pth
```

## Warum die Trennung notwendig ist

Der lokale Projektordner enthält sehr viele große Arbeitsdateien, während der
versionierte Quellcode nur wenige Megabyte umfasst. Git ist für
Versionsgeschichte und Zusammenarbeit gedacht, nicht als Sicherung großer
Mikroskopie-Datensätze. Dadurch bleiben Klonen, Vergleichen und Aktualisieren
des Codes schnell und zuverlässig.

Wenn später ausgewählte Modelle veröffentlicht werden sollen, sollten sie als
separat dokumentierte Release-Artefakte oder in einem geeigneten Datenarchiv
abgelegt werden. Git LFS ist nur für eine kleine, bewusst ausgewählte Zahl
großer Dateien sinnvoll, nicht für den vollständigen Arbeitsdatenbestand.
