# Instance-Annotationswerkzeug für Dataset142

[Zur Dokumentationsübersicht](../README.md)

Das lokale Werkzeug ordnet die bereits manuell annotierten Skeleton- und
Somapixel vollständigen Zellen zu. Eine Zell-ID ist nur innerhalb eines
Übersichtsbildes gültig. Die semantischen Original-Labels werden nie verändert.

## Start

```powershell
.\run_instance_annotation_tool.ps1
```

Die Oberfläche öffnet sich unter `http://127.0.0.1:8778/`. Standardmäßig werden
die drei MATS-Mosaike aus `Dataset140_matsOverview_finetune139` geladen.

Ein anderer kompatibler Datensatz kann angegeben werden:

```powershell
.\run_instance_annotation_tool.ps1 -DatasetRoot "C:\Pfad\Dataset" -OutputRoot "C:\Pfad\InstanceLabels"
```

Der Eingabedatensatz braucht `imagesTr/*_0000.tif` und passende
`labelsTr/*.tif` mit 0 = Hintergrund, 1 = Skeleton und 2 = Soma.

## Annotation

1. Mosaik auswählen.
2. `N` drücken und das Soma der neuen Zelle anklicken. Die komplette
   zusammenhängende Somakomponente wird dieser Zell-ID zugeordnet.
3. Mit `1` Skeletonpixel und mit `2` Somapixel malen. Der Pinsel verändert nur
   bereits vorhandene Pixel der gewählten semantischen Klasse.
4. `F` füllt eine zusammenhängende Skeleton- oder Somakomponente. An
   Kreuzungen sollte stattdessen gemalt werden, weil eine Komponente mehrere
   Zellen enthalten kann.
5. `V` ordnet Pixel zusätzlich einer zweiten Zelle zu.
6. `U` markiert nicht sicher lösbare Bereiche. Diese Pixel werden beim späteren
   Training ignoriert. Mit "Unsicher lösen" kann die Markierung entfernt werden.
7. `E` entfernt die Zuordnung der aktuell gewählten Zelle.

Die Pfeiltasten wechseln zwischen den 512-px-Bereichen. Zell-IDs bleiben dabei
über das gesamte Mosaik stabil. Jede Änderung wird sofort auf der Festplatte
gespeichert.

## Speicherung und Export

Der Arbeitsstand liegt unter:

`instance_annotation_dataset142/working`

"Trainingsdaten exportieren" schreibt eine unveränderliche Revision unter:

`instance_annotation_dataset142/exports/<mosaik>/revisions/rev_XXXXXX`

Jede Revision enthält:

- `instance_primary.tif`: primäre Zell-ID pro Vordergrundpixel
- `instance_secondary.tif`: zweite Zell-ID an echten Überlagerungen
- `ambiguous.tif`: Ignore-Maske
- `cells/cellXXXX/raw.tif`: Rohbildausschnitt der Zelle
- `cells/cellXXXX/label.tif`: semantisches 0/1/2-Label nur dieser Zelle
- getrennte Skeleton-, Soma-, Overlap- und Ignore-Masken
- `export_manifest.json`: Herkunft, Revision, Zellzahlen und Qualitätsprüfung

Die Originalbilder und Dataset140 bleiben unverändert.
