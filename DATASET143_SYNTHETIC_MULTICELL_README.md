# Dataset143: synthetische Mehrzellbilder aus Dataset141

Dataset143 erweitert die 2.759 freigegebenen Einzelzell-Crops aus Dataset141 um
reproduzierbar zusammengesetzte 512-x-512-Mikroskopiebilder mit zwei bis vier
Zellen. Das Ziel ist, Netz 141 an Nachbarschaft, Berührung und Überlappung zu
gewöhnen, ohne die wertvollen echten Einzelzellbilder zu ersetzen.

## Labelaufbau

- `labelsTr`: unveränderte semantische Klassen `0=Hintergrund`, `1=Skeleton`,
  `2=Soma`; Soma hat an Überlappungen Vorrang. Diese Labels nutzt das aktuelle
  nnU-Net-Fine-Tuning.
- `instance_primaryTr`: erste Zell-ID pro Pixel der synthetischen Bilder.
- `instance_secondaryTr`: zweite Zell-ID an echten Überlappungspixeln.
- `ambiguousTr`: derzeit überall 0; für spätere manuelle Unsicherheitsmarkierungen.

Die Instanzordner werden vom jetzigen semantischen Trainer ignoriert. Sie sind
bewusst schon vorhanden, damit dieselben synthetischen Szenen später für einen
zusätzlichen Instanz-/Embedding-Kopf verwendet werden können. Mit nur den drei
semantischen Ausgabekanälen kann das Netz nicht explizit lernen, dass zwei
berührende Skeletons zu verschiedenen Zellen gehören.

## Schutz der 1-px-Skeletons und gegen Data Leakage

- Nur exakte 90-Grad-Rotationen und Spiegelungen; keine interpolierende
  Verformung beim Erzeugen.
- Jede geeignete Einzelzelle wird genau einmal als Spender verwendet.
- Spender aus Fold-0-Training erzeugen ausschließlich Trainingsszenen;
  Fold-0-Validierungszellen ausschließlich Validierungsszenen.
- Die vollständige Herkunft und Transformation jeder Zelle steht in
  `synthetic_manifest.json` und `synthetic_manifest.csv`.
- Nicht geeignete Spender und der Grund stehen in `donor_audit.csv`.

## Erzeugen, ansehen und trainieren

```powershell
.\run_build_dataset143_synthetic_multicell.ps1
.\run_build_dataset143_synthetic_multicell.ps1 -Preprocess
.\run_training_dataset143_synthetic_multicell.ps1
.\run_training_dataset143_synthetic_multicell.ps1 -FullRun
```

Der erste Trainingsaufruf ist ein getrennter 20-Epochen-Techniktest. `-FullRun`
startet ein 200-Epochen-Fine-Tuning ab dem vollständigen finalen Checkpoint von
Netz 141 mit Lernrate 0,001. Architektur, 160-x-160-Trainingspatches,
Skeleton-Recall (Skeleton:Soma = 2:1), Deep Supervision und label-sichere
Augmentation bleiben gleich wie bei Netz 141.

Die HTML-Stichprobe liegt nach dem Aufbau unter:

`nnUNet_raw/Dataset143_dataset141_plus_synthetic_multicell/synthetic_multicell_review.html`

Vor einem langen Training sollten besonders unrealistische Helligkeitssprünge,
abgeschnittene Zellfortsätze und die erzeugten Berührungen in dieser Übersicht
kontrolliert werden.
