# Fünf Paper für die Verbesserung der Glioblastom-Segmentierungspipeline

> **Korrekturen nach unabhängiger Nachprüfung (2026-08-21):**
> 1. Der Bericht nannte an mehreren Stellen `geometry_rescue_gap = 110`. Der tatsächliche Wert
>    ist **60** — sowohl als Default in `run_skeleton_recall_to_evo_workflow.ps1:39` als auch
>    im protokollierten Lauf `20260817_TrogoCellStateTest/04_.../PBS/run_summary.json`.
>    Die Empfehlung, ihn zu senken, bleibt davon unberührt; nur der Ausgangswert war falsch.
> 2. Nachgemessen und bestätigt: 283 Validierungsfälle, β₀ Vorhersage 5.60 gegen GT 1.00,
>    44.9 % Ein-Komponenten-Anteil, 73.0 % Soma-Anbindung in-domain.
> 3. Nachgemessen und bestätigt: Hysterese auf NewImage2 senkt die Komponentenzahl von
>    1595 auf 578 und hebt die Soma-Anbindung von 48.3 % auf 83.1 % (Seed 0.50, Wachstum 0.02).
>    Der Bericht nannte 1582→565 und 43.4→82.1 %; die Abweichung stammt aus unterschiedlicher
>    Soma-Behandlung und ändert nichts an der Aussage.
> 4. Die Pipeline-Skripte verwenden bereits durchgängig 8-Konnektivität (14 von 14 Aufrufen
>    mit explizitem `structure`). Die Fragmentierung ist real, kein Zählartefakt.


Dieses Dokument fasst fünf Arbeiten zusammen, ordnet sie gegen die gemessenen Probleme des Projekts ein und benennt für jede, was sie leistet und was nicht. Alle Zahlen aus eigenen Messungen sind als solche markiert. Wo nur ein Abstract vorlag, steht es im jeweiligen Abschnitt.

Eine Vorbemerkung, die für alle fünf Abschnitte gilt: Keines dieser Paper adressiert die tatsächliche Hauptursache. Die Diagnose, die sich durch alle fünf kritischen Prüfungen zieht, lautet — das Netz versagt bereits **in-domain** topologisch (β₀ der Vorhersage 5.60 gegen GT 1.00, nur 44.9 % der Validierungs-Crops sind eine einzige Komponente), und auf Übersichtsbildern kommen Domänenverschiebung (Skalierung, Kontrast, Normalisierung) und Trainingsgeometrie (Einzelzell-Crops von 90×89 px bei Patch 96×96) hinzu. Die Paper liefern Messvorschriften, Nachbearbeitungsbausteine und eine gute Begründung für ein dickeres Label. Sie liefern keine Lösung für die Fragmentierung.

---

## Reihenfolge der Umsetzung

Sortiert nach Aufwand-Nutzen-Verhältnis. Alles oberhalb der Trennlinie kostet keine GPU-Zeit und arbeitet auf bereits vorhandenen Dateien.

| # | Schritt | Paper | Aufwand | Ohne Retraining | Erwarteter Nutzen |
|---|---|---|---|---|---|
| 1 | In-Domain-Topologie-Baseline aus den 283 vorhandenen fold-0-Validierungsvorhersagen (β₀^A, Ein-Komponenten-Anteil, Soma-Anbindung, Dice) | topo-pitfalls | 0.5 Tag | ja | **Hoch** — verschiebt die Diagnose. Gemessen: 5.60 / 44.9 % / 73.0 % gegen GT 1.00 / 100 % / 100 % |
| 2 | Wirkbereichs-Messung pro Übersichtsbild: Anteil Skelett in Multi-Soma-Komponenten, Waisenanteil, Abstandsverteilung Komponente↔Soma, Soma-Instanzstatistik | G-Cut | 0.5 Tag | ja | **Hoch** — entscheidet, ob Zuordnung überhaupt lohnt. Gemessen: nur 8.9 % strittig, 56.6 % Waisen |
| 3 | Hysterese-Sweep auf den vorhandenen `.npz` (Seed 0.5, Wachstum 0.30/0.20/0.10/0.05/0.02) mit Kontrollmetrik "Medianabstand der neuen Pixel zum Soma" | hysteresis-reconnect | 0.5–1 Tag | ja | **Hoch, aber bildabhängig** — auf NewImage2: 1582→565 Komponenten, 43.4→82.1 % am Soma. Auf anderen Bildern deutlich schwächer |
| 4 | Skalentest: Übersichtsbild mit Faktor 0.5/0.7/1.0/1.41/2.0 resampeln, vorhersagen, zurückskalieren | keins (aus Kritik) | 0.5 Tag | ja | **Hoch, ungetestet** — gemessener Skalenversatz 0.4062 vs 0.2875 µm/px, nnU-Net normiert nie |
| 5 | Normierungstest: Ganzbild- gegen Kachel-Inferenz (jede Kachel eigener Z-Score) | keins (aus Kritik) | 0.5 Tag | ja | **Mittel-hoch** — Kontrast Training 1.13 σ gegen Übersicht 0.46 σ |
| 6 | Augmentierungstest: Klasse-1-Komponenten vor/nach der echten Trainings-Transformkette zählen | skeleton-recall | 0.5 Tag | ja | **Mittel** — Proxy-Messung zeigt 2→14 Komponenten nach einer Rotation |
| 7 | Konnektivität dokumentieren (`topology_conventions.py`, β₀^A-Notation, ein Absatz ins README) | topo-pitfalls | 2 h | ja | **Niedrig, aber Pflicht** — Berichtshygiene; Faktor 38 zwischen 4- und 8-Konnektivität |
| 8 | Metrik-Umstellung: CCQ mit Toleranz 0/1/2/3 px, Komponentenzahl ab 5 px, Soma-Anbindung als Tripel | oner + skeleton-recall | 1 Tag | ja | **Mittel** — ohne sie ist kein Fortschritt belegbar |
| 9 | Vorhandene Brückenkette entschärfen statt erweitern (geometry_rescue_gap 60→30–40) | hysteresis-reconnect | 1–2 Tage | ja | **Mittel** — 96 % der gebauten Brücken stammen aus dem ungeprüften Geometry-Rescue |
| 10 | Union-Find-Fusionsschutz: keine Verbindung zweier Komponenten, die je ein Soma enthalten | hysteresis-reconnect | 1 Tag | ja | **Mittel** — schon bei 2 px Toleranz 310 Multi-Soma-Labels |
| — | **Ab hier: Annotation oder GPU nötig** | | | | |
| 11 | Instanz-Ground-Truth auf FOV-/Übersichtsebene (3–8 Kacheln, alle Zellen) | alle | 2–5 Tage | ja (manuell) | **Sehr hoch, indirekt** — ohne das ist keine der übrigen Maßnahmen falsifizierbar |
| 12 | Dataset139 mit dilatiertem Klasse-1-Label (3–5 px), gleicher Trainer, Skeletonize nach Inferenz | skeleton-recall | 1 Tag + 4 h GPU | nein | **Mittel** — Faktor 2–4 weniger Fragmente plausibel, Verschmelzungsrisiko real |
| 13 | Trainer-Aufräumung: Identitätsabbildung in `TubedSkeletonTargetTransform` dokumentieren/ersetzen | skeleton-recall | 1 h | nein (mit 12) | **Null Qualität** — nur Ehrlichkeit und CPU-Ersparnis |
| 14 | DS-Gewichte der groben Stufen (1/4, 1/8) auf 0 | skeleton-recall | 1 h + 4 h GPU | nein | **Niedrig** — Defekt real (2.5→18.6 Komponenten), Beitrag der Stufen gering |
| 15 | Rekonnektionsfilter überarbeiten (Winkel kalibrieren, Bresenham-Veto umformulieren, Gegenseitigkeit als Score) | hysteresis-reconnect | 2–3 Tage | ja | **Begrenzt** — harte Obergrenze gemessen: 25–53 % Anbindung selbst im Bestfall |
| 16 | Kreuzungsauflösung an Verzweigungsclustern | G-Cut + hysteresis | 2–3 Tage | ja | **Unsicher** — Grad-4-Knoten existieren im Netz-Skelett kaum |
| 17 | G-Cut-LP mit eigenem GOF-Prior aus SWC-Tracings | G-Cut | 1–2 Wochen | ja | **Klein bis null** — Wirkbereich 8.9 %, Konkurrenz erreicht schon 87.3 % |
| 18 | Kontextdatensatz mit ignore_label und großen Patches | skeleton-recall | 1–2 Wochen + GPU | nein | **Potenziell hoch, teuer** — erfordert Loss-Umbau (`NotImplementedError`) |
| — | **Verworfen** | | | | |
| ✗ | Threshold-Sweep als Fragmentierungslösung | topo-pitfalls | — | — | Gemessen negativ: Komponenten steigen um 20–40 % |
| ✗ | Snake-Annotationskorrektur (Öner) | oner | — | — | Prämisse widerlegt: Versatz 0.148 px Median |
| ✗ | Snake als Lückenschließer | oner | — | — | Ersetzt globales Dijkstra durch lokalen Abstieg mit <1 px Reichweite |
| ✗ | Gröbere künftige Tracings | oner | — | — | Beschädigt das wertvollste Asset ohne funktionierende Snakes |
| ✗ | tube_radius = 0 | skeleton-recall | — | — | Null Effekt Klasse 1, Schaden Klasse 2, trotzdem Trainingskosten |
| ✗ | Nearest-Neighbour-Tube-Propagation | skeleton-recall | — | — | Konfligierende Gradienten, adressiert falschen Fehlermodus |
| ✗ | Betti-Matching | topo-pitfalls | — | — | Bibliothek fehlt, bei 1-px-Persistenz auf Rauschniveau |

---

## 1. Pitfalls of Topology-Aware Image Segmentation

*Berger, Lux, Weers, Menten, Rueckert, Paetzold. IPMI 2025. arXiv:2412.14619*
Volltext gelesen (PDF via pdftotext, 51583 Zeichen, alle Abschnitte und Tabellen).

### Was das Paper sagt

Dies ist kein Verfahrenspaper, sondern eine Evaluationskritik. Die These: Vergleiche topologiebewusster Segmentierungsmethoden sind systematisch verzerrt, und zwar durch drei Fallstricke — unpassende Konnektivitätswahl, übersehene topologische Artefakte im Ground Truth, und ungeeignete oder falsch aggregierte Metriken.

**Der Konnektivitätsteil ist der für dieses Projekt entscheidende.** Um ein diskretes Binärbild überhaupt in einen topologischen Raum zu übersetzen, muss festgelegt werden, ob diagonal benachbarte Pixel zur selben Komponente gehören. Das Paper definiert für ein D-dimensionales Bild zwei Nachbarschaften: *direct connectivity* mit 2·D Nachbarn (in 2D also 4, Pixel teilen eine Kante) und *all connectivity* mit 3^D − 1 Nachbarn (in 2D also 8, inklusive der Diagonalen, deren Ränder sich nur in einem Eckpunkt schneiden).

Entscheidend ist: Damit der Jordansche Kurvensatz gilt, **müssen** Vordergrund und Hintergrund entgegengesetzte Konnektivität benutzen. Es gibt daher genau zwei zulässige Einstellungen, die das Paper mit einem Buchstaben benennt:

- **A** := all(8) für Vordergrund, direct(4) für Hintergrund
- **D** := direct(4) für Vordergrund, all(8) für Hintergrund

Wer persistente Homologie benutzt, wählt über die Komplex-Konstruktion implizit eine davon: Die V-Konstruktion entspricht D, die T-Konstruktion entspricht A. Meist wird das nicht berichtet — von allen zitierten Vorarbeiten dokumentiert genau eine ihre Wahl, und zwar die für ihren Datensatz semantisch ungünstige.

**Das Suszeptibilitätsmaß (Gleichung 1)** ist das einzige "Verfahren" des Papers und in wenigen Zeilen scipy nachzubauen. Man labelt dasselbe Ground Truth zweimal — einmal mit D-, einmal mit A-Konnektivität — und bildet daraus zwei Partitionen P_D und P_A. Dann ist die Konnektivitäts-Suszeptibilität für die 0-te Betti-Zahl:

```
β0^{D vs A}_err = | β0(P_D) − β0(P_A) |
```

β0 zählt die Vordergrund-Zusammenhangskomponenten. Weil β0_err eine echte Metrik ist, spielt die Reihenfolge keine Rolle. Analog werden VOI^{D vs A} und ARE^{D vs A} gebildet. Hohe Werte sagen voraus, dass die Methoden-Scores unter D und A stark auseinanderlaufen.

Die gemessenen Zahlen sind drastisch. Auf DRIVE (Retinagefäße) hat dasselbe Label unter A-Konnektivität **132** Vordergrundkomponenten und unter D-Konnektivität **18850** — Faktor 143. Die Suszeptibilität beträgt dort 467.95. Auf CREMI und Roads ist Dimension 0 unauffällig (99.0 % bzw. 99.5 % Übereinstimmung), dafür ist Dimension 1 anfällig. MSSEG2-3D ist in beiden Dimensionen unauffällig und damit ein Beispiel für einen Datensatz, bei dem die Wahl egal ist.

Die Konsequenz für Methodenvergleiche wird auf CREMI gemessen: Sechs Verlustfunktionen (Dice, clDice, HuTopo, BettiMatching, Mosin, TopoGraph) wurden jeweils unter beiden Konnektivitäten trainiert **und** evaluiert. Die Rangkorrelationen zwischen A und D sind durchweg negativ — Spearman ρ von −0.37 bis −0.85, im Mittel −0.63 über die sechs topologischen und distributionellen Metriken. Die Rangfolge kehrt sich also um. Die Interpretation der Autoren: Methoden, die im Training konnektivitätsunabhängig sind (Dice, Mosin, clDice), schneiden auch unter der semantisch falschen Konnektivität passabel ab; konnektivitätsabhängige Methoden spielen ihre Stärke nur unter der richtigen Wahl aus. Gute Werte unter der ungünstigen Konnektivität sagen nichts über die Leistung unter der richtigen voraus.

**Der Artefaktteil** definiert topologische Artefakte als Merkmale, die im Label existieren, aber keine oder eine falsche semantische Bedeutung tragen. Drei Ursachen: Konnektivitätsartefakte (entstehen zwangsläufig, weil weder A noch D alle Probleme löst — auf DRIVE erzwingt die nötige FG-8-Konnektivität eine BG-4-Konnektivität, die ein einziges intervaskuläres Gebiet in zahllose Komponenten zerlegt), Label-Rausch-Artefakte (Einzelpixel-Annotationsfehler, auf Dice vernachlässigbar, auf die topologische Repräsentation dramatisch) und Auflösungsartefakte. Der Nachweis läuft über Training und Evaluation einmal mit Original-GT und einmal nach Entfernung aller Komponenten ≤ 5 px. Auf DRIVE sinkt dadurch β1_err um 42.9 % und BM1_err um 28.3 %, während Dice sich um 0.38 % ändert. Bis zu 43 % der gemessenen topologischen Fehler waren also Artefakte.

*(Hinweis zur Genauigkeit: Der Fließtext auf S. 9 vertauscht die Werte −29 % und −43 % zwischen β1_err und BM1_err gegenüber Tabelle 4. Nachgerechnet über die absolute Differenzzeile stimmt die Tabelle, nicht der Fließtext.)*

**Der Metrikteil** warnt vor drei Dingen. Erstens: Pixelweise Metriken sind für Werte < 1 topologisch nicht interpretierbar. Zweitens: Distributionelle Metriken (VOI, ARE) verschränken topologische und volumetrische Fehler irreversibel und hängen selbst an der Konnektivitätswahl. Drittens — und das ist die schärfste Aussage — topologische Fehler dürfen **niemals über Dimensionen hinweg aggregiert** werden. Auf CREMI korrelieren BM0_D und BM1_D mit ρ = −0.77, sind also antikorreliert; das in mindestens sieben zitierten Arbeiten übliche Summieren zu `BM_err = β0_err + β1_err` zerstört die Aussagekraft. Empfohlen wird stattdessen, die Konnektivität als Superskript an jede Metrik zu hängen: β0^A gegen β0^D.

### Warum das für dich relevant ist

Ein 1-px-Bresenham-Skelett ist der extremste denkbare Fall von Konnektivitätsanfälligkeit. Das ist auf Dataset138 nachgemessen:

- β0^A (8-Konnektivität, Klasse 1): **3442** Komponenten gesamt, 2.44 pro Crop
- β0^D (4-Konnektivität, Klasse 1): **131471** Komponenten gesamt, 93.11 pro Crop
- Verhältnis 2.62 %, also **Faktor 38**. Suszeptibilität 90.67 pro 90×89-px-Crop

Zum Vergleich: DRIVE erreicht 467.95 auf 565×584-px-Bildern. Pro Fläche normiert ist Dataset138 deutlich anfälliger. Die Ursache ist verifiziert: `rasterize_div10_training_masks.py` zeichnet die Skelette mit klassischem Bresenham (Zeilen 151–170), bei dem x- und y-Schritt in derselben Iteration feuern können. Das GT-Skelett enthält also echte Diagonalschritte und ist strikt 8-zusammenhängend. Unter 4-Konnektivität zerfällt es in Einzelpixel. 8-Konnektivität ist hier keine Präferenz, sondern zwingend.

Zwei weitere Befunde aus derselben Messung sind wichtiger als der Faktor 38:

**Das GT hat das Anbindungsproblem nicht.** In 1411 von 1412 Crops (99.9 %) bilden Skelett und Soma unter 8-Konnektivität *eine einzige* Komponente. Die auf Übersichtsbildern gemessenen 6–60 % Soma-Anbindung sind folglich kein GT-Artefakt, sondern ein echtes Vorhersage- oder Nachbearbeitungsproblem.

**Die richtige Zielgröße ist nicht β0(Klasse 1) = 1.** Im GT hat die Skelettklasse allein im Mittel 2.44 Komponenten pro Crop, weil das Soma sie naturgemäß trennt — jede Primärneurite ist eine eigene Komponente. Die semantisch korrekte Zielgröße ist **β0^A(Klasse 1 ∪ Klasse 2) = 1** pro Zelle. Die bereits erhobene Kennzahl "Anteil der Skelettpixel mit Soma-Anschluss" ist damit exakt die richtige Hauptmetrik; der GT-Referenzwert dafür ist 99.9 %.

β1^A liegt bei 0.168 Schleifen pro Crop — Dimension 1 ist auf Einzelzell-Crops nahezu bedeutungslos und wird erst auf Übersichtsbildern durch kreuzende Neuriten relevant.

### So wendest du es an

1. **Zuerst messen — In-Domain-Baseline aus den vorhandenen Validierungsvorhersagen.** Über `nnUNet_results\Dataset138_cleanSingleCell_soma_skeleton_recrop\nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x__nnUNetPlans__2d\fold_0\validation` gegen `nnUNet_raw\...\labelsTr`. Kennzahlen je Fall, nie über Dimensionen aggregiert: β0^A(Klasse 1 ∪ Klasse 2), Anteil Fälle mit β0 = 1, Anteil somaverbundener Skelettpixel, Skelettpixelzahl, Dice Klasse 1. **Diese Messung ist bereits gelaufen: 5.60 / 44.9 % / 73.0 % / 245 px / 0.719 gegen GT 1.00 / 100 % / 100 % / 205 px.** Diese Tabelle ist ab jetzt die Vergleichsbasis jedes weiteren Experiments.

2. **Konnektivität festschreiben, nicht umbauen.** Die bestehende Kette benutzt bereits durchgängig 8-Konnektivität für den Vordergrund — geprüft in `postprocess_net129_no_loss.py` (483, 492, 1214, 1469), `compare_cell_assignment_methods.py` (587, 590, 728, 731) und `prepare_somas_for_assignment.py` über `CONNECTIVITY_8`. Die einzige hintergrundbezogene Operation, `ndi.binary_fill_holes` in `prepare_somas_for_assignment.py` Zeile 191, hat als Default bereits `generate_binary_structure(2,1)`, ist also 4-konnektiv und Jordan-konsistent. Es gibt nichts zu reparieren. Lege trotzdem `topology_conventions.py` mit `CONNECTIVITY_FG = np.ones((3,3))` und `CONNECTIVITY_BG = ndi.generate_binary_structure(2,1)` an und stelle die Aufrufe darauf um — reine Absicherung gegen künftiges Auseinanderlaufen. Verhalten ändert sich dabei nicht.

3. **Einen Absatz dokumentieren, keinen wiederkehrenden QC-Report.** In `DATASET137_DIV10_BUILD_README.md` und `PROJEKT_HANDOFF_2026-08-20.md`: das GT-Skelett ist wegen des Bresenham-Rasterizers strikt 8-zusammenhängend, unter 4-Konnektivität zerfällt es um Faktor 38, deshalb ist 8-Konnektivität für den Vordergrund zwingend und wird an jede Metrik als β0^A notiert. Der Wert ist eine Konstante des Rasterizers, kein Messwert, der sich ändert — ein wiederkehrender Suszeptibilitäts-Report würde eine Konstante messen.

4. **Berichtsnotation umstellen.** In allen HTML-Ausgaben (`make_skeleton_recall_comparison_html.py`, `make_strict_soma_qc_html.py`) und im Handoff: "β0^A (FG 8-konnektiv, BG 4-konnektiv)" statt "Komponentenzahl". Ohne diese Angabe kann sich jede berichtete Komponentenzahl um Faktor 38 unterscheiden.

5. **Pseudo-Dice als Steuergröße degradieren, aber nicht abschaffen.** nnU-Net braucht ihn für die Checkpoint-Auswahl. In den Berichten wird er zur volumetrischen Partnermetrik neben β0^A(Klasse 1 ∪ 2) und dem Soma-Anbindungsanteil. β0 und β1 niemals summieren. β1 gar nicht erst als Steuergröße verwenden — bei 1-px-Strukturen ist es fast blind: Verdicken erzeugt keine Löcher, sondern füllt sie, und Rausch-Speckles erzeugen ebenfalls keine.

6. **Die 307 GT-Komponenten ≤ 5 px sichten, nicht löschen.** 8.9 % der Skelettkomponenten in Dataset138 fallen in diese Größenklasse — genau die, die das Paper auf DRIVE als Artefakt entfernt und auf CREMI ausdrücklich nicht. Für ein 1-px-Skelett mit echten kurzen Neuritenstummeln ist der CREMI-Fall der wahrscheinlichere. Als Galerie ausgeben (`make_cell_gallery.py` existiert), dann entscheiden. Niedrige Priorität — das erklärt weder 1000–12000 Vorhersagekomponenten noch 44.9 % Ein-Komponenten-Fälle.

### Einschränkungen

**Das Paper reduziert kein einziges Fragment.** Es ist eine Benchmarking-Kritik und liefert die Messvorschrift, nicht die Lösung. Wer eine Verbesserung der Segmentierung erwartet, wird enttäuscht.

**Der Threshold-Sweep, der aus dem Paper abgeleitet wurde, ist empirisch widerlegt.** Ausgeführt auf den vorhandenen `.npz`: Bild 20X_LSM_CF (4152×4152): t=0.50 → 31248 Skelettpixel, 2206 Komponenten, 17.2 % am Soma; t=0.05 → 57415 Pixel, 3099 Komponenten (+40 %), 27.1 % am Soma. Bild CF_30XLP_10x_1_Exp100ms_binningon: t=0.50 → 2205 Komponenten, 4.7 % am Soma; t=0.05 → 2716 Komponenten, 4.1 % am Soma, also **schlechter**. Die Lücken liegen nicht knapp unter der Schwelle — dort ist p1 praktisch null. Als dokumentiertes Negativergebnis abschließen. *(Beachte: das steht in Spannung zum Hysterese-Befund in Abschnitt 4 — der Unterschied ist die komponentenbasierte Wachstumslogik statt eines globalen Schwellwerts, und die Bildabhängigkeit ist erheblich.)*

**Die Empfehlung "kleine Komponenten entfernen" ist nicht übertragbar.** Die Autoren sagen selbst, dass sie auf DRIVE und Roads hilft und auf CREMI essenzielle Information zerstören würde. Der CREMI-Fall ist hier der zutreffende.

**Statistisch schwache Basis.** Alle Rangkorrelationen beruhen auf n = 6 Methoden. ρ = −0.77 ist bei n = 6 grenzwertig signifikant. Die qualitative Aussage ("Rangfolge ist instabil") ist robust, die konkreten Zahlen sind es nicht. Das Konnektivitätsexperiment lief nur auf CREMI, das Artefaktexperiment nur auf DRIVE — keine Kreuzvalidierung.

**Kein Bezug zu Mikroskopie, Zellen oder 1-px-Labels.** Evaluiert wird auf DRIVE, CREMI und Roads. DRIVE ist die nächste Analogie, aber DRIVE-GT ist eine *dicke* Gefäßmaske. Der Fall, dass das Label selbst bereits die Mittellinie ist, kommt im Paper nirgends vor. Die Übertragung ist eine Ableitung, keine Aussage der Autoren.

**Betti-Matching wird empfohlen, ist aber nicht verfügbar.** Im `.venv` ist keine Topologie-Bibliothek installiert (weder gudhi noch ripser noch cripser). Bei 1-px-Strukturen haben persistenzbasierte Verfahren ohnehin Barcodes mit Persistenz 0 bis 1 Pixel — jedes Merkmal liegt auf Rauschniveau. Reine β0/β1-Fehler sind mit scipy sofort berechenbar, Betti-Matching nicht und lohnt hier nicht.

**Ein unauflösbarer Zielkonflikt bleibt.** Das Paper fordert "eine Konnektivität pro Datensatz nach der Semantik der Struktur". Hier sind zwei Semantiken gegenläufig: Kontinuität *eines* Neuriten verlangt 8-Konnektivität, Trennung *zweier* sich kreuzender Neuriten verlangt 4. Das Paper bietet dafür nichts an. Hinzu kommt: Tumor-Mikrotuben verbinden Glioblastomzellen untereinander — die Zielgröße β0^A(Klasse 1 ∪ 2) = 1 pro Zelle ist ein Artefakt der Einzelzell-Kuratierung, keine Eigenschaft realer Kulturen. Der korrekte Zielwert auf Übersichtsbildern ist ohne Instanz-Annotation unbekannt.

**Realistische Nutzenerwartung:** Null Verbesserung der Segmentierung. Der Wert liegt in Punkt 1 der Umsetzungsliste — der In-Domain-Baseline. Sie zeigt, dass das Modell schon in perfekter Domäne topologisch versagt, und verschiebt damit die Priorität weg von der Nachbearbeitung.

---

## 2. Adjusting the Ground Truth Annotations for Connectivity-Based Learning to Delineate

*Öner, Kozinski, Citraro, Fua. IEEE TMI 41(12):3675–3685, 2022. arXiv:2112.02781*
Volltext gelesen (ar5iv-HTML) plus vollständiger Quellcode des Repos.

### Was das Paper sagt

Das Paper greift eine Annahme an, die bei dünnen Strukturen fast immer falsch ist: dass die manuelle Annotation pixelgenau auf der wahren Mittellinie liegt. Öner et al. argumentieren, dass menschliche Tracings von Neuriten und Gefäßen topologisch korrekt, geometrisch aber um wenige Voxel versetzt sind — und dass genau dieser Versatz bei 1-Voxel-Strukturen den Trainingsverlust dominiert.

Die Begründung ist einfach: Standardtraining minimiert Θ* = argmin_Θ Σ L(ŷ_i, y_i) mit ŷ_i als angeblich korrekter Annotation. Bei einer 1-px-Struktur ist ŷ_i im Wesentlichen eine Nullmenge — verschiebt man sie um 1 px, sinkt der Dice-Überlapp auf nahe 0. Das Netz kann diesen Fehler nicht durch bessere Geometrie beheben, weil der Versatz im Label steckt, nicht in der Vorhersage. Also lernt es zu hedgen: es verbreitert, verwischt und unterbricht die Vorhersage, weil das den erwarteten Verlust über inkonsistent versetzte Labels minimiert. Genau das erzeugt nach dieser These fragmentierte Skelette.

**Die Reformulierung** behandelt die Annotation als Variable statt als Konstante:

```
Θ*, C* = argmin_{Θ, C} Σ_i [ L(c_i, y_i) + R(c_i) ]
```

C_i = (V_i, E_i) ist der Annotationsgraph von Bild i, c_i ∈ R^{k×d} der Stapel der Knotenkoordinaten. **Die Kantenmenge E_i wird nie verändert** — nur Koordinaten bewegen sich. Damit bleibt die Topologie per Konstruktion erhalten; keine Komponente kann zerreißen oder verschmelzen. Das ist der zentrale Trick.

**Rendern des Graphen als abgeschnittene Distanzkarte.** Für jedes Voxel q wird der euklidische Abstand zum nächstgelegenen Punkt auf einem Kanten-Liniensegment berechnet und bei d_max gekappt:

```
D(c)[q] = min{ δ(c, q), d_max }
δ(c, q) = min_{(u,v) ∈ E} min_{0 ≤ φ ≤ 1} || φ·c_u + (1−φ)·c_v − q ||₂
```

φ parametrisiert den Punkt auf der Strecke von c_u nach c_v. Das ist stetig und nach c differenzierbar — deshalb kann durch das Rendering hindurch abgeleitet werden. Im Code läuft das lokal auf Crops von 32³ um jeden Knoten.

**Anders als bei nnU-Net gibt das Netz keine Klassenwahrscheinlichkeit aus, sondern regressiert die Distanzkarte selbst.** Der Datenterm ist MSE: `loss = torch.pow(pred_dmap − snake_dm, 2).mean()`. Die binäre Mittellinie entsteht zur Auswertung durch Schwellwert 2 auf der vorhergesagten Distanzkarte plus Skeletonisierung. Das ist der größte strukturelle Unterschied zum vorliegenden Setup.

**Die Snake-Energie** hält Form und Topologie:

```
R(c) = α · Σ_{(u,v) ∈ E} || c_u − c_v ||²  +  β · Σ_{(u,v,w) ∈ T} || c_u − 2c_v + c_w ||²
```

Der erste Term (Feder, α) bestraft Kantenlängen und hält Knoten beisammen; Nebenwirkung ist Schrumpfen. Der zweite Term (Elastizität, β) ist die zweite Differenz über Knotentripel, in denen v genau zwei Nachbarn hat, und bestraft Krümmung. Als quadratische Form: R(c) = ½·cᵀAc mit dünnbesetzter Steifigkeitsmatrix A. Deren Aufbau ist direkt nachprogrammierbar — pro Kante (u,v): A[u,u] += α, A[v,v] += α, A[u,v] −= α, A[v,u] −= α; pro Knoten v mit exakt zwei Nachbarn u,w: A[v,v] += 4β, A[u,u] += β, A[w,w] += β, A[v,u] und A[u,v] −= 2β, A[v,w] und A[w,v] −= 2β, A[u,w] und A[w,u] += β. Verzweigungsknoten und Endpunkte bekommen keinen β-Term.

**Das Update ist ein impliziter Euler-Schritt:** (A + γI)c^{t+1} = γc^t − ∂L/∂c. Mit stepsz = 1/γ wird daraus c^{t+1} = (stepsz·A + I)^{-1}·(c^t − stepsz·g_ext(c^t)). Die Matrix C := (stepsz·A + I)^{-1} wird einmal pro Graph vorberechnet. Pro Trainingsiteration laufen T = 10 solcher Schritte.

**SnakeFast** (die empfohlene Variante) bewegt den Snake nicht mit dem vollen Datenterm, sondern mit einer billigen Ersatzenergie S(c,y) = Σ_v (y * G)[c_v]. Weil y eine Distanzkarte ist (klein auf der Mittellinie), ist Gradientenabstieg auf S die Bewegung "bergab in Richtung des vorhergesagten Zentrums". **Beim Portieren auf eine Wahrscheinlichkeitskarte (groß auf der Mittellinie) muss das Vorzeichen invertiert werden**, sonst laufen die Snakes vom Skelett weg.

Kritisch für das Vertrauen in das Verfahren: Die Snakes werden in **jedem** forward() neu aus der Original-Annotation initialisiert. Die Korrektur akkumuliert nicht über Epochen; sie startet jedes Mal beim menschlichen Tracing und läuft 10 Schritte. Die Drift ist damit auf etwa 2 Voxel pro Iteration beschränkt, der Einzugsbereich durch d_max = 15 gegeben.

**Parameter aus Paper und Code:** α = 1e-2, β = 1e-3 (Paper-Optimum, Tabelle IV). *Achtung:* die veröffentlichte `main.config` enthält α = 1e-4, β = 1e-2 — also nicht die als bester Wert markierte Kombination. γ = 10 bzw. stepsz = 0.1, nsteps = 10, fltrstdev = 0.5, extgradfac = 2.0, dmax = 15, cropsz = 32³, maxedgelength = 5, ours_start = 0 (kein Warm-up).

**Ergebnisse.** Auf Brain (Zweiphotonen, 14 Stapel): UNet mit Originalannotation erreicht APLS 80.3, mit SnakeFast 91.1 — **+10.8 Punkte**. Die Kernaussage: dieser Sprung ist größer als der von UNet-OrigAnnot (80.3) zu DRU-OrigAnnot (84.3), also größer als der Gewinn durch den Architekturwechsel. Auf absichtlich vergröberten Annotationen (nur gerade Linien zwischen Verzweigungspunkten) springt die Completeness von 67.6 auf 87.0 und APLS von 46.5 auf 66.8. Bei perfekten synthetischen Annotationen ist SnakeFast gleichauf mit dem Baseline-Training — die Methode schadet also nicht.

### Warum das für dich relevant ist

Die Ausgangsbeobachtung des Papers beschreibt das Projekt exakt: 1-px-GT, Pseudo-Dice als Steuergröße, fragmentierte Vorhersagen. Wenn die These stimmt, wäre das die Ursache.

**Sie stimmt hier nicht.** Die Vorab-Diagnose wurde ausgeführt — sie brauchte keine Neuinferenz, weil die 283 fold-0-Validierungsvorhersagen bereits vorliegen:

- Kohärenter Versatz zwischen vorhergesagter Mittellinie und Annotation: **Median 0.148 px** (p90 2.43), mittlerer Betrag 0.482 px
- Correctness bei Toleranz 0/1/3 px: 0.702 / 0.857 / 0.941
- Completeness bei Toleranz 0/1/3 px: 0.752 / 0.845 / 0.922
- Halbbreite der Distanztransformierten in der **Vorhersage**: Median 1.00 px, exakt wie das GT

Das Paper adressiert Versätze von "a few voxels" mit einem Einzugsbereich von 15 Voxeln. Ein Versatz von 0.15 px ist Diskretisierungsrauschen. Und der postulierte Mechanismus tritt nicht auf: Das Netz verbreitert nicht (Halbbreite 1.00), und im Definitionsbereich fragmentiert es kaum (GT Median 2 Komponenten, Prediction 4). Die Ursachenkette ist an beiden Enden gerissen.

Was bleibt, ist trotzdem wertvoll — nämlich zwei Nebenbefunde, die aus der Beschäftigung mit dem Paper entstanden sind:

**Die Metrikeinsicht.** Das Paper verwendet bewusst nicht Dice als Hauptmetrik, sondern CCQ mit Toleranz d = 3, APLS und TLTS. Für 1-px-Zentrallinien ist toleranzbehaftete Precision/Recall die einzige Metrikfamilie, die nicht degeneriert. Auch die tatsächliche Ausgangszahl ist eine andere als angenommen: 0.6662 ist der EMA-Pseudo-Dice aus der Trainingsschleife; der echte fold-0-Validierungs-Dice steht in `validation\summary.json` und beträgt **0.7190** für Klasse 1 und 0.9392 für Klasse 2. Außerdem: n_pred 244.6 gegen n_ref 205.4 — das Netz sagt 19 % mehr Skelettpixel voraus als annotiert.

**Der eigentliche Annotationsversatz kommt von der Augmentierung, nicht vom Menschen.** nnU-Net resampelt die Segmentierung in `SpatialTransform` mit Nearest-Neighbour (`nnUNetTrainer.py` ab Zeile 760: p_rotation 0.2, Rotation ±180°, p_scaling 0.2, Skalierung 0.7–1.4). Gemessen an 120 Dataset138-Labels: Klasse-1-Komponenten **Median 2 → 14** nach einer einzigen Rotation (Mittelwert 29.8), 2 → 3.5 nach reiner Skalierung. Self-Dice nach Rotation und Rückrotation: **0.591**. Allein die Rasterung deckelt den erreichbaren Dice einer 1-px-Klasse also bei etwa 0.6 — dieselbe Größenordnung wie der beklagte Pseudo-Dice. Jeder fünfte Trainingspatch trägt ein in ~14 Stücke zersprungenes Ziel.

### So wendest du es an

1. **Der Diagnoseschritt ist bereits erledigt, das Ergebnis ist negativ.** Das konsolidierte Skript liegt im Scratchpad unter `diagnose_annotation_offset.py` und kann unverändert nach `C:\Ole\20260721_CellClassification_v2\diagnose_annotation_offset.py` übernommen werden. Halte den Befund fest, damit er nicht in drei Wochen erneut diskutiert wird. Wichtiger Hinweis zur Methodik: Die Diagnose **muss** auf dem val-Split laufen. Der Vorschlag "auf 30–50 der 1412 Crops vorhersagen" hätte laut `splits_final.json` in 80 % der Ziehungen Trainingsfälle getroffen (1129 train / 283 val) — gegen memorierte Labels ist der Versatz konstruktionsbedingt null.

2. **CCQ-Metrik übernehmen, APLS nicht.** Der Code steckt in der Funktion `a_b_offset` des Diagnoseskripts und braucht 30 Minuten, um in `summarize_nnunet_training_runs.py` oder ein eigenes `eval_ccq.py` zu wandern. Toleranzstufen 0/1/2/3 px. APLS ist hier ungeeignet: Es vergleicht Pfadlängen zwischen Knotenpaaren, und bei Einzelzell-Crops mit Median 92 Skelettpixeln und 2 Komponenten ist die Metrik nahezu entartet.

3. **Augmentierungsschaden messen, dann beheben.** 200 echte augmentierte Trainingspatches aus `nnUNetDataLoaderSkeletonRecall` (`custom_trainers\skeleton_recall\nnUNetTrainerSkeletonRecallCells.py`, Zeile 105) abgreifen und Klasse-1-Komponenten pro Patch zählen. Der scipy-Proxy sagt Median 2 → 14; die Zahl aus dem echten `grid_sample`-Pfad muss das bestätigen, bevor etwas geändert wird.

4. **Dann eine von zwei kleinen Änderungen.** (a) Rotation auf Vielfache von 90° beschränken — zusammen mit dem ohnehin aktiven Mirroring ist das die volle Dihedralgruppe und für ein 1-px-Skelett exakt labelerhaltend. (b) Nach `SpatialTransform` und vor `TubedSkeletonTargetTransform` eine Reparatur einhängen: binäres Closing der Klasse 1 mit 3×3 plus `skeletonize`, um Treppenlücken zu schließen. Beides sind wenige Zeilen im vorhandenen Trainer, ein Lauf kostet ~4 Stunden.

5. **Bewerten gegen die Baseline aus Abschnitt 1, plus Fragmentstatistik auf einem festen Übersichtsbild.** Eine Verbesserung im Crop-Definitionsbereich sagt noch nichts über die Übersicht.

### Einschränkungen

**Die Kernprämisse ist für dieses Projekt gemessen widerlegt.** Versatz 0.148 px Median gegen "a few voxels" im Paper. Damit fallen sämtliche Snake-bezogenen Maßnahmen ersatzlos weg.

**Auch strukturell würde es nicht funktionieren.** Der Einzugsbereich existiert hier nicht: dmax = 15 funktioniert im Paper, weil das Netz eine Distanzkarte regressiert, die 15 Voxel neben der Mittellinie noch informativ ist. p1 aus nnU-Net fällt innerhalb von 1–2 px auf 0 ab. Mit fltrstdev = 0.5 ist die effektive Reichweite der externen Kraft deutlich unter 1 px — ein Snake, der 2 px daneben liegt, spürt nichts. Sigma hochzudrehen verschmilzt benachbarte parallele Neuriten. Der Vorschlag "y = dmax·(1−p1) mit dmax = 6" repariert das nicht: er skaliert eine Funktion, die außerhalb von 2 px konstant 6 und damit gradientenfrei ist.

**Die Voraussetzung "Annotation liegt als Graph vor" gilt nur für 18 % der Daten.** Richtig ist, dass `rasterize_div10_training_masks.py` SWC liest und die Parent-Spalte die Kantenmenge ist. Aber von 1412 Fällen in Dataset138 haben nur 260 ein div10-Präfix; die übrigen 1152 stammen aus Dataset136 und haben keine Graph-Quelle. Für 82 % müsste man skeleton→graph rekonstruieren — genau der verlustbehaftete Schritt, der angeblich entfällt, und zwar an Kreuzungen und Verzweigungen.

**Der Trainer hat die Toleranz bereits eingebaut** (wenn auch, siehe Abschnitt 3, nicht für Klasse 1). Der marginale Zugewinn wäre kleiner als vermutet.

**2D-Kultur verletzt die Strukturannahmen.** Das Paper behandelt eine Vordergrundklasse, baumartig, in 3D räumlich getrennt. Hier kreuzen sich Neuriten verschiedener Zellen in der Projektionsebene, und es gibt eine zweite Flächenklasse. Ein Snake, der von ∇(p1*G) getrieben wird, kennt keine Zellidentität — an einer Kreuzung werden die Graphen zweier Zellen auf denselben Grat gezogen. Im Paper ist das ein nicht untersuchter Randfall, hier wäre es der Normalfall.

**Der Snake als Lückenschließer dupliziert vorhandene, stärkere Maschinerie.** `postprocess_net129_no_loss.py` baut in `build_cost_image` (Zeile 426) bereits ein Kostenbild aus P(Skelett), Rohbildevidenz und optionaler Ridge-Evidenz und läuft mit `route_through_array` (Zeile 858), also Dijkstra, plus `direction_cosine` (Zeile 539) mit Gate, Detour-Ratio und Score. **Dijkstra ist auf diesem Kostenfeld global optimal; ein Snake ist lokaler Gradientenabstieg auf einer geglätteten Version desselben Feldes mit Reichweite unter 1 px.** Er kann nur verlieren. Das Argument "hohes β bevorzugt an Kreuzungen die Fortsetzungsrichtung" ist als `direction_cosine` bereits implementiert.

**Gröbere künftige Tracings sind gefährlich.** Tabelle III zeigt, dass Snakes grobe Annotationen retten — nicht, dass grobe Annotationen ohne funktionierende Snakes tragfähig sind. Die vorhandenen 1412 kuratierten Zellen mit gemessenen 0.15 px Versatz sind das wertvollste Asset des Projekts und derzeit das Einzige, worauf sich alle Vergleiche stützen können.

**Das Kostenmodell steht kopf.** Neutraining kostet gemessen 14.5 s/Epoche, 1000 Epochen also ~4 Stunden. Teuer wäre die Neuimplementierung des Snake-Stacks (renderDistanceMap, snake.py, gradImSnake, Knoten-Resampling, QC) mit 3–5 Tagen, der zusätzliche Regressionskopf mit 1–2 Wochen. Der Engpass ist nicht die GPU, sondern die fehlende belastbare Zielmetrik.

**Realistische Nutzenerwartung:** Für die Snake-Methode praktisch null, und das ist gemessen, nicht geschätzt. Ein Dataset139-Experiment mit verschobenen SWC-Knoten würde bei 3–5 Tagen Aufwand einen Rauschunterschied von wenigen Zehntel CCQ-Punkten liefern, mit nicht vernachlässigbarem Verschlechterungsrisiko durch das vom Paper selbst benannte Schrumpfen der Terminaläste (Median nur 92 Skelettpixel pro Zelle). Der reale Ertrag liegt in den zwei Nebenbefunden: der Metrikdisziplin und dem Augmentierungsschaden.

---

## 3. Skeleton Recall Loss

*Kirchhoff, Rokuss, Roy, Kovacs, Ulrich, Wald, Zenk, Vollmuth, Kleesiek, Isensee, Maier-Hein. ECCV 2024. arXiv:2404.03010*
Volltext gelesen (arXiv-HTML). PDF-Abruf scheiterte am Größenlimit; alle Zahlen über mehrere unabhängige Abfragen konsistent bestätigt.

### Was das Paper sagt

Skeleton Recall Loss ist ein Zusatzterm zum nnU-Net-Standardloss, der die Konnektivität dünner röhrenförmiger Strukturen erhalten soll, ohne die teure differenzierbare Soft-Skeletonisierung von clDice zu brauchen. Der Trick: Die Skelettierung läuft einmal auf dem **Ground Truth** auf der CPU beim Data Loading, nicht differenzierbar auf der GPU über der Prädiktion.

**Algorithmus 1 — Tubed Skeletonization**, vier Schritte auf dem harten Multiklassenlabel Y:

1. **Binarisieren:** Y_bin ← Y > 0. Alle Vordergrundklassen werden zu *einer* Maske verschmolzen.
2. **Skelettieren:** Y_skel ← skeletonize(Y_bin). In 2D Zhang & Suen (1984), in 3D Lee et al. (1994) — also genau `skimage.morphology.skeletonize`. Ergebnis ist 1 px breit.
3. **Dilatieren zum Tube:** Y_skel ← dilate(Y_skel, diamond(radius=2)). Der Diamond-Kernel mit Radius 2 umfasst in 2D alle Pixel mit |dx|+|dy| ≤ 2, also 13 Pixel; das Skelett wird effektiv 5 px breit. Begründung im Paper: die Dilatation vergrößert die effektive Fläche für die Loss-Berechnung und stabilisiert sie durch Einbeziehung von mehr Pixeln.
4. **Klassenzuweisung durch Multiplikation:** Y_mc-skel ← Y_skel · Y. Der binäre Tube wird elementweise mit dem Original-Multiklassenlabel multipliziert.

**Schritt 4 ist der kritische.** Er schneidet den Tube auf den GT-Vordergrund zurück. Im Paper-Setting ist das harmlos, weil das GT (Gefäße, Risse, Straßen) deutlich dicker als 5 px ist — der Tube liegt vollständig *innerhalb* des GT, die Multiplikation vergibt nur Labels. **Ist das GT dünner als der Tube, wird die Dilatation vollständig rückgängig gemacht.**

**Algorithmus 2 — die Loss.** Pro Vordergrundklasse c (Hintergrund ausgeschlossen, do_bg=False):

```
L_mc-skel[c] = − Σ_Pixel( Y_mc-skel[c] · p[c] ) / Σ_Pixel( Y_mc-skel[c] )
L_SkelRecall = mean über c ∈ {1..K−1}
L_gesamt = L_Dice + L_CE + w · L_SkelRecall
```

Zähler ist die Summe der vorhergesagten Wahrscheinlichkeiten der richtigen Klasse an allen Tube-Pixeln (weiche True Positives), Nenner die Anzahl der Tube-Pixel (TP + FN). Das ist exakt Recall in weicher Form. Getestet wurde w ∈ {0.1, 1.0}; welches w für welchen Datensatz benutzt wurde, berichtet das Paper nicht.

**Kein Präzisionsterm.** Das Paper begründet das nicht explizit (gezielt gesucht, nicht gefunden). Die strukturelle Begründung folgt aus der Konstruktion: Das Skelett ist eine echte Teilmenge des GT. Ein Pixel, das korrekt Vordergrund vorhersagt, kann außerhalb des Tubes liegen (am Rand des dicken Objekts). Ein Präzisionsterm würde diese korrekten Pixel bestrafen. Die Bestrafung von Übersegmentierung bleibt vollständig bei Dice und CE.

**Der Gradient ist konstant:** ∂L/∂p[c][j] = −1/Σ(Y_mc-skel[c]) für Tube-Pixel, 0 sonst — unabhängig vom aktuellen Prädiktionswert. Ein Pixel bei p = 0.99 bekommt denselben Schub wie eines bei p = 0.01.

**Ergebnisse** auf fünf Datensätzen (Roads, DRIVE, Cracks, ToothFairy, TopCoW), nnU-Net-Backbone, Default → Skeleton Recall: Roads Dice 78.99 → 79.25, Betti-0 5.769 → 4.846. DRIVE 80.87 → 80.99, Betti-0 **57.00 → 38.75**. Cracks 94.59 → 94.88. ToothFairy 71.80 → **74.42**, clDice 89.16 → 92.05. TopCoW binär 93.55 → 93.72, multiklasse 85.36 → 86.59.

Die Dice-Gewinne sind mit +0.12 bis +2.62 Punkten klein. Der robusteste Befund ist nicht Genauigkeit, sondern **Ressourceneffizienz**: +8 % Trainingszeit und +2 % VRAM gegenüber +88 % und +52 % bei clDice. Bei 13 Klassen läuft clDice auf einer A100-40GB in OOM, SRL bleibt konstant. Und SRL ist die erste multiklassenfähige Loss für dünne Strukturen.

### Warum das für dich relevant ist

Der Trainer `nnUNetTrainerSkeletonRecallCellsSkeleton2xSoma1x` ist eine paper-treue Implementierung — geprüft in `custom_trainers\skeleton_recall\nnUNetTrainerSkeletonRecallCells.py`, Zeilen 55–58: skeletonize, dilation mit diamond(2), Multiplikation mit label_map. Die Frage ist, was dieser Code bei einem 1-px-GT tatsächlich tut.

**Gemessen auf 60 Labels aus Dataset138:** Der Tube für Klasse 1 umfasst **8552 Pixel gegenüber 8556 GT-Pixeln der Klasse 1 — Faktor 1.00.** Von 48948 dilatierten Tube-Pixeln werden 67.4 % durch die Multiplikation in Schritt 4 wieder verworfen. Für Klasse 2 (Soma) funktioniert der Mechanismus wie vorgesehen: nur 23.8 % der Soma-Pixel landen im Tube, also eine echte Medialachsen-Röhre.

**Konsequenz:** Der Skeleton-Recall-Term für Klasse 1 ist mathematisch identisch mit einem gewöhnlichen Soft-Recall-Loss auf `label_map == 1`. Kein Schlauch, keine Toleranz, kein Topologiesignal. Der Trainername ist in diesem Projekt irreführend.

**Wichtige Präzisierung:** Der Term ist deshalb **nicht wirkungslos.** In `20280812_newTest2` liegen drei Predictions desselben Bildes vor:

| Variante | Skelettpixel | Komponenten | Größte | Am Soma |
|---|---|---|---|---|
| Baseline (ohne SRL) | 16004 | 1932 | 203 | 9.3 % |
| SRL w=1, gleiche Gewichte | 29454 | 1509 | 295 | 47.0 % |
| SRL 2x Skelett / 1x Soma | 32147 | 1472 | 334 | 49.3 % |

SRL verändert das Verhalten massiv, und die 2x-Gewichtung legt nochmal zu. Ein unbestrafter Recall-Term treibt die Sensitivität der dünnen Klasse — das erklärt +84 % Skelettpixel plausibel. **Ob die zusätzlichen Pixel richtig sind, ist mangels Ground Truth auf Übersichtsbildern unbekannt.** Genau diese FP-Inflation beschreibt das Replikationspaper arXiv:2508.11374 für einen Loss ohne Präzisionsterm.

**Zwei weitere gemessene Defekte im Ziel:**

*Deep Supervision.* `DownsampleSkeletonForDSTransform` (Zeile 68–103) samplet mit `nearest-exact` herunter. Gemessen auf 100 Labels explodieren die Klasse-1-Komponenten pro Fall auf Skala 1/2 von **2.5 auf 18.6** (Faktor 7.4), auf 1/4 auf 1.5 px pro Komponente. Die DS-Gewichte dieser Stufen liegen nach Normalisierung bei ~33 % und ~17 %. Das Netz wird also mit erheblichem Gewicht auf ein zerhacktes Punktwolken-Ziel trainiert.

*Augmentierung vor der Skelettierung.* `TubedSkeletonTargetTransform` wird **nach** der räumlichen Augmentierung eingehängt. Das 1-px-Label ist beim Loss-Aufruf also bereits zerhackt (siehe Abschnitt 2). Das ist die stärkste Begründung für ein dickeres Label — stärker als das Argument "mehr Gradientenmasse".

### So wendest du es an

1. **Den Befund zuerst nachvollziehen.** Die Skripte `check_tube.py` und `check_ds.py` liegen im Scratchpad. Sie zeigen Faktor 1.00 für Klasse 1 und 23.8 % Restanteil für Klasse 2.

2. **Trainer-Aufräumung, ohne Erwartung an Qualität.** In `TubedSkeletonTargetTransform.apply` dokumentieren oder ersetzen: für Klasse 1 liefert skeletonize + dilation + Multiplikation zu 99.95 % exakt `label_map == 1` zurück — eine teure Identitätsabbildung mit ~15500 skeletonize-Aufrufen pro Epoche. Entweder als Kommentar festhalten oder für Klasse 1 durch `(label_map == 1)` ersetzen. Bitweise dasselbe Ziel bei geringerer CPU-Last. **Das ist eine Ehrlichkeitsmaßnahme, keine Verbesserung.**

3. **Dickes Label als das eine lohnende Neutraining.** Nicht `rasterize_div10_training_masks.py` ändern — es deckt laut `rasterization_report.csv` nur 400 von 1412 Fällen ab; ~1012 stammen aus Dataset136 und sind nicht aus SWC reproduzierbar. Stattdessen ein kleines Skript, das die vorhandenen `labelsTr` von Dataset138 liest: `skel5 = binary_dilation(label==1, diamond(2))`, dann `neu = 2 wo label==2, sonst 1 wo skel5, sonst 0`. Die Soma-Priorität muss erhalten bleiben (der Rasterizer setzt heute erst Skelett, dann Soma — das Soma gewinnt). Der 8-px-Rand der Crops reicht für eine 2-px-Dilatation.

4. **Mit dem unveränderten paper-treuen Trainer trainieren** (radius 2, w = 1.0, class_weights = None) und die Vorhersage in der Nachbearbeitung mit `skimage.morphology.skeletonize` auf 1 px ausdünnen. Begründung, in der richtigen Reihenfolge: (a) ein 5-px-Label überlebt nnU-Nets Resampling-Augmentierung und das DS-Downsampling, ein 1-px-Label nicht; (b) Skeleton Recall wird dadurch überhaupt erst funktionsfähig — dicke Maske → skeletonize → Schlauch hat Platz, genau das Setting des Originalpapers; (c) Dice wird gegenüber 1-px-Versatz stabil. Bei 0.29–0.41 µm/px entsprechen 5 px etwa 1.4–2.0 µm, also grob der physikalischen Breite eines Tumor-Mikrotubus — das Label wird dadurch nicht unehrlicher.

5. **Vorher den Round-Trip auf verrauschten Daten testen, nicht nur auf sauberem GT.** Gemessen auf 148 Dataset138-Labels (GT → dilate(diamond2) → skeletonize): Längenverhältnis Median 0.970, Median-Positionsfehler 0.00 px, Verzweigungspunkte unverändert — aber bereits 9 % der isolierten Crops verlieren Komponenten durch Verschmelzung. Bei Übersichtsdichte ist es dramatischer: dilatiert man das *prädizierte* Skelett von NewImage2 mit diamond(2), sinkt die Komponentenzahl von 1595 auf 674 — **58 % der Objekte verschmelzen bei 5 px Breite**. Auf sauberen, isolierten Labels ist der Round-Trip nahezu verlustfrei; auf einer dichten Prediction ist er ungetestet und riskant.

6. **DS-Gewichte der groben Stufen auf 0 setzen** (Einzeiler in `_build_loss`), bevor man `max_pool` probiert. Bei Patch 96 ist Skala 1/8 gerade 12×12 px und trägt für eine 1-px-Struktur keine Information. Der `max_pool`-Fix tauscht nur eine Verzerrung gegen eine andere: Klasse-1-Flächenanteil `nearest` 1.45/1.44/1.38/1.46 % gegen `max_pool` 1.45/4.59/9.18/**18.33** % auf den Skalen 1, 1/2, 1/4, 1/8. Auf Skala 1/8 wäre also 18 % des Crops "Skelett" — das Ziel ist dann keine dünne Struktur mehr.

7. **Kontrolllauf mit w = 0.1** (`nnUNetTrainerSkeletonRecallCellsW01`, Zeile 663) als Teil derselben Trainingsbatterie, nicht als Einzelmaßnahme. Zu prüfen ist die Precision der Skelettklasse gegen den aktuellen Lauf. Aktuell ist `skeleton_recall_weight = 1.0` mit `class_weights (2.0, 1.0)`, also effektiv 0.667 für Klasse 1 statt der 0.5 im Paper-Mittelwert.

### Einschränkungen

**Das Paper zeigt nichts für 1-px-GT.** Alle fünf Datensätze haben dicke Masken, aus denen ein Skelett *abgeleitet* wird. Der zentrale Mechanismus setzt voraus, dass das GT dicker ist als der Tube.

**Kein Ablationsexperiment zum Tube-Radius.** Der Wert 2 wird gesetzt, nicht begründet, nicht variiert.

**Ein Replikationspaper widerspricht der Kernbehauptung.** "Does the Skeleton-Recall Loss Really Work?" (arXiv:2508.11374, nicht peer-reviewed) leitet den konstanten Gradienten analytisch her und berichtet: auf tubulären Datensätzen signifikante Verbesserung in nur 3 von 15 Metriken, signifikante Verschlechterung in 3; die False-Positive-Rate steigt auf allen Datensätzen. Auf nicht-tubulären Datensätzen verschlechtert SRL durchgängig. Die Autoren zeigen zudem, dass die Maskentransformation ein Ergebnis liefert, das dem GT zu ähnlich ist — genau der hier quantitativ nachgemessene Effekt.

**Konnektivität kann nur innerhalb des Patches erzwungen werden.** Bei 512×512 (Roads) deckt ein Patch eine Straßenkreuzung ab. Bei 96×96 auf einem 35000×35000-Bild sieht das Netz nie einen Neuriten über seine volle Länge. Aber: **die gemessene Fragmentierung ist lokal, nicht langreichweitig.** Medianer Komponentendurchmesser 6 px; `binary_closing` mit r = 1/2/3 senkt die Komponentenzahl nur von 1595 auf 1093/943/870. Ein größeres Patch adressiert diesen Fehlermodus nicht direkt.

**Patch 512 hat zwei harte Blocker.** Erstens gibt es kein dichtes GT für Übersichtskacheln — alle Tracings liegen als Einzelzell-Crops vor (Median 241×249 px). Eine 512×512-Kachel enthielte viele nicht annotierte Nachbarzellen, die als Hintergrund gelernt würden. Zweitens wirft `nnUNetTrainerSkeletonRecallCells` bei `ignore_label` explizit `NotImplementedError` ("This trainer is intentionally restricted to fully annotated labels"). Ohne Ignore-Label-Unterstützung ist die Empfehlung nicht umsetzbar.

**Verworfene Varianten:** `radius = 0` ist für Klasse 1 mathematisch identisch zum Ist-Zustand und zerstört für Klasse 2 den einzigen funktionierenden Tube — bei vollen Trainingskosten. Die Nearest-Neighbour-Propagation der Klassenlabels auf den Tube belohnt Vorhersagen bis 2 px neben dem wahren Skelett ohne Präzisionsstrafe, während Dice und CE dieselben Pixel bestrafen: konfligierende Gradienten, und sie adressiert einen Fehlermodus (2-px-Versatz), der gemessen nicht auftritt.

**Ungenannte, vermutlich größere Ursachen.** (a) *Skalenfehlanpassung:* Die Trainings-TIFFs tragen XResolution 1.0, `nnUNetPlans.json` enthält `original_median_spacing_after_transp [999.0, 1.0, 1.0]` — nnU-Net normiert die Skala **nie**. Die DIV10-Tracings entstanden bei 0.4062 µm/px, `NewImage2_0000.tif` trägt 0.2875 µm/px. Unkorrigierter Faktor 1.41, bei den 40x-Bildern mehr. Für eine 1–2 px breite Zielstruktur ist ein Skalenfehler von 41 % ein Fehler erster Ordnung — ohne Neutraining testbar und nie getestet. (b) *Domänenverschiebung durch Dataset138 selbst:* `build_dataset138_clean_single_cell_from_dataset137.py` schneidet jede Zelle auf ihre Bounding-Box plus 8 px und prüft explizit auf helle Nachbarn. Im Training heißt Hintergrund also "verifiziert leer" bei genau einer isolierten Zelle; bei der Inferenz enthält jedes 96×96-Fenster fremde, nicht annotierte Zellen. Das Netz wurde darauf trainiert, alles außer der einen kuratierten Zelle zu unterdrücken. **Dataset138 hat dieses Problem gegenüber Dataset137 verschärft.** (c) *Labelrauschen:* In 150 zufälligen Crops liegt der Median bei 4.3 % hellen Pixeln ohne Label in 2-px-Umgebung, aber 22 von 150 (15 %) haben über 25 % ihres hellen Signals als Hintergrund gelabelt. Der eingebaute Detektor hat nur 25 von 1412 Fällen markiert, weil er auf kompakte helle Nachbarn zielt und dünne fremde Neuriten übersieht.

**Biologische Annahmeverletzung.** Die Loss-Kommentare im Code nennen Klasse 1 selbst "Skeleton/TM". Tumor-Mikrotuben sind per Definition interzelluläre Verbindungen und bilden ein Netzwerk mit Zyklen, keine zellweisen Bäume. Damit ist "weniger Löcher ist besser" kein gültiges Ziel, und die Prämisse "jeder Neurit gehört zu genau einem Soma" ist für einen Teil der Strukturen ill-posed. Das ist eine Definitionsfrage, die von der Biologie entschieden werden muss.

**Realistische Nutzenerwartung:** Die Trainer-Aufräumung bringt null Qualität. Der DS-Fix bringt wenig, weil die groben Stufen bei Patch 96 ohnehin kaum beitragen. Das dicke Label ist die einzige Maßnahme mit plausibler Aussicht auf substantiell weniger Fragmente — Prognose bewusst zurückhaltend: Faktor 2–4 weniger Skelettkomponenten und deutlich stabilerer Dice sind plausibel; 100 % Soma-Anbindung nicht, weil ein Teil der Lücken keine Bildevidenz hat. Es kann scheitern, wenn die Verdickung an Kreuzungen mehr Falschverbindungen erzeugt, als sie echte Lücken schließt. **Ohne Übersichts-GT ist das nicht entscheidbar.**

---

## 4. MS-LSDNet und geometrische Skelett-Rekonnektion

*Du, Zhang, Song, Bao, Zhang, Wu, Liu. Computers in Biology and Medicine 153:106416, 2023*

> **⚠ Nur der Abstract war zugänglich.** Das Paper ist strikt closed access. Geprüft: OpenAlex (`oa_status: closed`, keine oa_url, `abstract_inverted_index: null`), Semantic Scholar (kein openAccessPdf), kein arXiv-Preprint, kein PMC, kein Eintrag in IA Scholar, kein Code-Repository auffindbar. ScienceDirect, ResearchGate, core.ac.uk: HTTP 403 bzw. "Request PDF". Gelesen wurde ausschließlich der wortgetreue Abstract über NCBI eutils (PMID 36586230).
>
> **Alles Konkrete unten stammt aus zwei frei zugänglichen, methodisch vergleichbaren Arbeiten, die im Volltext gelesen wurden, und ist NICHT das Verfahren von Du et al.** Zahlen, die Suchmaschinen-Zusammenfassungen der Verlagsseite nennen (Accuracy 98.08 % DRIVE, 97.14 % STARE, 98.94 % HRF, 0.013 s Laufzeit), konnten in keiner tatsächlich gelesenen Quelle bestätigt werden und sind als unbelegt zu behandeln. Zur Plausibilität: 98.08 % Accuracy auf DRIVE liegt deutlich über dem üblichen Bereich von 95.5–97 %.

### Was das Paper sagt

Der Abstract nennt genau drei Stufen und sonst nichts Technisches: (1) ein *multiscale linear structure detection network* (MS-LSDNet), das feine Gefäße durch Lernen reichhaltiger hierarchischer Merkmalstypen besser detektiert; (2) eine *adaptive hysteresis threshold method*, die beim Binarisieren der Wahrscheinlichkeitskarte die Konnektivität erhält; (3) ein *vascular tree structure reconstruction algorithm based on a geometric skeleton*, der gebrochene Segmente verbindet. Es gibt keine Endpunktdefinition, keine Richtungsschätzung, kein Suchkriterium, keine Verbindungsregel, keinen Schwellwert, keine Ablationstabelle.

Die übertragbare These lautet: **Fragmentierung ist zu einem erheblichen Teil ein Binarisierungs- und Postprocessing-Problem, nicht nur ein Netzproblem** — der Schritt von der Wahrscheinlichkeitskarte zur Maske ist der Ort, an dem Konnektivität verloren geht.

**Hysterese-Schwellwertbildung**, Standarddefinition plus Condurache & Aach (MVA 2005, Volltext gelesen). Gegeben eine Wahrscheinlichkeitskarte p(x):

- Starke Menge: S_high = { x : p(x) ≥ T_high }
- Schwache Menge: S_low = { x : p(x) ≥ T_low }, mit T_low < T_high
- Ergebnis: Vereinigung aller 8-zusammenhängenden Komponenten von S_low, die mindestens ein Pixel aus S_high enthalten

T_high liefert nur hochkonfidente Pixel (viele FN), T_low praktisch alle Vordergrundpixel (viele FP). Das Vorwissen "das Objekt ist zusammenhängend" wählt aus S_low genau die Pixel aus, die an einen hochkonfidenten Kern angebunden sind. Rauschen über T_low ohne Kernanbindung fällt heraus. Derselbe Mechanismus wie bei Canny, nur auf einer Wahrscheinlichkeitskarte. In Python: `skimage.filters.apply_hysteresis_threshold(p, low, high)`, intern mit voller 8-Konnektivität.

Condurache & Aach geben zwei Wege zur adaptiven Bestimmung. Formal über Hypothesentests: T_high aus der Hintergrunddichte über ein Signifikanzniveau α (∫_{T_high}^∞ p(x|ω_b) dx = α), T_low aus der Vordergrunddichte über eine Fehlerwahrscheinlichkeit ε; beide Dichten per Parzen-Fenster mit Gauß- oder Exponentialkern geschätzt, Stichproben aus einer Otsu-Trennung. Praktisch über eine Perzentilregel: T_high als Perzentil, das einen Anteil a der Bildfläche einschließt (dort a = 6 %), T_low bei 100 % − b (Optimum b = 87 %, also oberste 13 %). Verhältnis der Flächenanteile ≈ 2.2:1. Ergebnis am Optimum: 3.21 % FP bei 77.72 % TP, AUC 0.9643 — zum Vergleich der zweite menschliche Beobachter: 2.29 % FP bei 77.73 % TP.

**Geometrische Endpunkt-Rekonnektion**, vollständig spezifiziert nach Dulau et al. (ICCV 2023 Workshop CVAMD, Volltext gelesen). Die Pipeline heißt VNR und besteht aus zwei Prozeduren.

Punkttypen auf dem Skelett (8-Nachbarschaft): *Endpunkt* = genau ein weißer Nachbar, *Kreuzpunkt* = drei oder mehr, *glatter Punkt* = genau zwei. Wichtiger empirischer Befund: **die anatomisch korrekten Anschlusspunkte liegen überwiegend auf glatten Punkten der Hauptstruktur**, nicht auf Endpunkten oder Kreuzpunkten. Reine Endpunkt-zu-Endpunkt-Verfahren erzeugen deshalb entweder sehr wenige oder sehr lange, falsche Verbindungen.

*Prozedur 1, Gap Filling.* Komponenten unter 5 px werden vorab entfernt. Kandidaten sind **gegenseitig nächste Paare** (A ist Bs nächster Punkt UND B ist As nächster Punkt), sowohl Endpunkt-Endpunkt als auch Endpunkt-Kreuzpunkt. Das ist der entscheidende Unterschied zu "jeder Endpunkt nimmt seine k nächsten Nachbarn". Gehören beide Punkte zur selben Komponente, wird verworfen. Sonst: in der *kleineren* Komponente den zum Kandidaten c nächstgelegenen weiteren Endpunkt/Kreuzpunkt q suchen und den Winkel bei c zwischen Partnerpunkt p und q messen. **Richtungskriterium: höchstens 45° Abweichung von einer Geraden**, also Winkel(p,c,q) ∈ [135°, 180°]; äquivalent mit der nach außen zeigenden Tangente t = (c−q)/|c−q| und u = (p−c)/|p−c|: ⟨t,u⟩ ≥ 0.7071. Die Richtungsschätzung ist also eine Sehne über das Fragment — kein Fitting, keine Ableitung. **Okkupationstest:** entlang der Bresenham-Verbindung die Pixelwerte auslesen; liegt bereits Gefäß darauf, wird nicht verbunden. Der Verbindungsweg ist eine **Gerade, ausdrücklich kein Spline** — Begründung der Autoren: Splines hängen extrem von den Kontrollpunkten ab, eine Verschiebung um 1 px verzerrt stark, bei kurzen Wegen nähern sie sich ohnehin einer Geraden, bei langen entstehen abrupte Winkel. **Selbstskalierende Lückengrenze:** der Rekonnektionsweg muss kürzer sein als die Skelettlänge der kleineren Komponente — ersetzt eine manuell gesetzte maximale Distanz. Die Dicke wird über einen 5×5-Kernel vom Endpunkt der kleineren Komponente entlang der Linie propagiert.

*Prozedur 2, Rebranch or Remove.* Die größte Komponente ist "main". Für jede kleinere Komponente wird die Gerade q→e über den Endpunkt e hinaus verlängert; liegt ein Hauptpunkt auf der Verlängerung, wird verbunden, sonst wird die Komponente **entfernt**. Hier wird der Winkel bewusst ignoriert.

**Die Gewinnaufteilung** ist für das Zielpaper nicht beantwortbar (keine Ablation), aber aus VNR quantitativ eindeutig: Postprocessing bringt praktisch nichts in Dice und alles in Topologie. Über vier Datensätze: clDICE-Netz Dice 0.839 → 0.840 bei **174 → 1** Komponenten; UNetBN Dice 0.844 → 0.845 bei **114 → 1**. Die Autoren erklären das: Rekonnektionswege bestehen aus sehr wenigen Pixeln, also kann der Dice nicht steigen — aber genau diese Pixel machen abgetrennte Äste überhaupt erst messbar. Nebenbefund: das topologiebewusste clDICE-Netz produzierte **mehr** Komponenten (174) als das rein pixelorientierte UNetBN (114). Ein Topologie-Loss garantiert keine Konnektivität.

VNR verwarf außerdem ein gelerntes Nachbearbeitungsnetz (DVAE): es reduzierte auf im Mittel 39 Komponenten, verdickte aber die Gefäße deutlich. Verdickung tauscht Konnektivität gegen Messbarkeit — eine direkte Warnung für ein 1-px-Projekt.

### Warum das für dich relevant ist

Die Kernthese trifft hier zu, aber nur schwach — und stark bildabhängig. Gemessen auf `20260817_ObjectiveTest\03_dataset138_predictions`:

| Bild | argmax | Hysterese (0.5/0.05) |
|---|---|---|
| 10X_WF_4 | 88.921 px, 3964 Komp., 6.1 % am Soma | 131.990 px, 2012 Komp., 9.3 % |
| 10X_CF_4 | 163.960 px, 9005 Komp., 24.0 % am Soma | 231.800 px, 5446 Komp., 31.5 % |

Faktor ~2 bei den Komponenten und +3 bis +7 Prozentpunkte Anbindung. Real, aber keine Lösung. **Und der Komponentengewinn ist größtenteils Buchhaltung:** zählt man nur Komponenten ab 5 px, ändert sich beim CF-Bild fast nichts (4702 → 4686), beim WF-Bild 2166 → 1793. Die Hysterese schluckt überwiegend Sub-5-px-Staub.

Auf einem anderen Bild (`20280812_newTest2\08_predictions_dataset138\NewImage2`, 5040×5056, 236 Somata) fällt der Sweep deutlich besser aus, weil dort komponentenbasiertes Wachstum statt eines globalen Schwellwerts benutzt wurde:

| Schwelle | Komponenten | Median | Am Soma |
|---|---|---|---|
| 0.5 (argmax) | 1582 | 6 px | 43.4 % |
| 0.5/0.20 | 943 | — | 63.0 % |
| 0.5/0.10 | 763 | — | 71.6 % |
| 0.5/0.05 | 639 | — | 77.7 % |
| 0.5/0.02 | **565** | **44 px** | **82.1 %** |

**Und die Kontrollrechnung fällt positiv aus:** die bei 0.02 hinzugekommenen 27.518 Pixel liegen im Median 48.8 px vom nächsten Soma entfernt (Seed-Pixel: 45.7 px), nur 7.6 % liegen näher als 3 px. Der Gewinn ist echtes longitudinales Lückenschließen entlang der Neuriten, nicht triviales Anwachsen am Soma.

**Die harte Obergrenze des gesamten Rekonnektionsprogramms ist gemessen.** Simuliert wurde der gutmütigste denkbare Rekonnektor — Dilatation von Skelett+Soma um r Pixel mit unbegrenzter Verkettung, was jedes gerichtete Verfahren mit gleicher Lückenweite dominiert. Soma-angebundene Skelettpixel: WF 6.1 % (r=0) → 13.6 % (Gap 6) → **25.1 %** (Gap 24). CF 24.0 % → 42.1 % → **52.8 %**. Selbst im physikalisch unerreichbaren Bestfall bleiben 47–75 % unzuordenbar. Der Grund: **Median-Abstand einer Skelettkomponente zum nächsten Soma beträgt 147–153 px.** Nur 11.4 % der Komponenten (WF) bzw. 18.6 % (CF) liegen überhaupt im Wirkbereich der aktuellen Gaps von 22–30 px. Kein Postprocessing überbrückt 150 px korrekt.

Dieselbe Simulation zeigt die Kehrseite: schon bei Gap 2 px enthalten 310 Labels (WF) mehr als eine Soma-Komponente; bei Gap 24 px hängen im größten Label 122 verschiedene Somata zusammen.

### So wendest du es an

1. **Zuerst messen, pro Übersichtsbild.** Ein Diagnoseskript, das ausgibt: (a) Komponentenzahl gesamt und ab 5 px, (b) Größenverteilung, (c) Anteil Skelettpixel mit Soma-Anbindung, (d) **Verteilung des Abstands jeder Komponente zum nächsten Soma** — das ist die entscheidende, bisher fehlende Zahl, (e) Anzahl und Größenverteilung der Soma-Komponenten, (f) die Dilatations-Obergrenzenkurve: Anbindung als Funktion der erlaubten Lückenweite 0/2/4/6/10/16/24 px, zusammen mit der Zahl fusionierter Somata. Punkt (f) beantwortet in einer Stunde, ob sich die Rekonnektionsarbeit lohnt.

2. **Kein neues Hysterese-Skript schreiben.** `postprocess_overview_v3.py` enthält in Zeile 62–70 und in `recover_skeleton_from_probabilities` (ab Zeile 243) bereits exakt die Hysterese samt Distanzdeckel: `SKELETON_LOW_PROBABILITY = 0.12`, `SOMA_EXCLUSION_PROBABILITY = 0.50`, `MAX_PROBABILITY_RECOVERY_DISTANCE_PX = 12`, `ndi.binary_propagation` mit 3×3-Struktur. Die reale Lücke ist, dass diese Funktion nie in die aktuelle Kette (`postprocess_net129_no_loss.py`) portiert wurde.

3. **Schwellwerte als Sweep bestimmen, nicht per Perzentilregel.** T_low ∈ {0.20, 0.15, 0.10, 0.05, 0.02} bei T_high = 0.5, Auswahl nach den Metriken aus Schritt 1. **Die Somamaske unverändert aus dem argmax übernehmen** und Skelettkandidaten explizit außerhalb halten — sonst wachsen sie in Somata hinein und verändern Zähler *und* Nenner der Zielmetrik. Kandidat für den neuen Arbeitspunkt nach den Messungen: Seed 0.5, Wachstum 0.05 bis 0.02.

4. **Immer die Kontrollmetrik mitführen:** Medianabstand der *neu hinzugekommenen* Pixel zum nächsten Soma. Ohne sie ist "Anteil Skelett am Soma" trivial manipulierbar — er steigt auch, wenn man nur Konfetti wegwirft.

5. **Speicher für die großen Bilder lösen.** Die `.npz` sind hier float32, nicht float16: bei 35000×35000 sind das 3 × 1.225e9 × 4 Byte = **14.7 GB** allein für das Wahrscheinlichkeitsarray, das `np.load` vollständig dekomprimiert. Ein globales `ndi.distance_transform_edt` liefert zusätzlich float64 (9.8 GB) und mit `return_indices` zwei int64-Arrays (19.6 GB) — das sprengt 61 GB. Zwei Wege: den Distanzdeckel als N-fache maskierte Dilatation (bool-Arrays, N = 12 Durchläufe) oder gekachelt mit Überlappung ≥ N und Zusammenführung über Komponenten-IDs. Laufzeit ist kein Problem: Hysterese auf 86 MP dauert ~8 s, `skeletonize` ~2 s.

6. **Die vorhandene Brückenkette zurückdrehen, nicht erweitern.** `postprocess_net129_no_loss.py` hat bereits `--endpoint-gap`, `--min-direction-cosine`, `--min-path-probability` plus wahrscheinlichkeits- und bildgeführtes Routing. Der dokumentierte Lauf fuhr mit endpoint_gap 45, geometry_rescue_gap 60, min_path_probability 0.04: von 72.778 Endpixeln sind **25.881 (36 %) hinzugefügt, davon 24.812 (96 %) aus dem Geometry-Rescue** — nie validiert. Auf dem neuen Arbeitspunkt entschärfen: geometry_rescue_gap auf 30–40, endpoint_gap auf 20–25, min_path_probability deutlich hoch. Abnahme: Brückenanteil sinkt deutlich, Komponentenzahl steigt nicht, Anbindung fällt nicht.

7. **Union-Find-Fusionsschutz einbauen.** Vor allen Feinheiten: **keine Verbindung darf zwei Komponenten vereinigen, die bereits je ein Soma enthalten.** Das ist billiger, robuster und wirksamer als Voronoiregionen und bildet den Unterschied zwischen "ein Gefäßbaum pro Bild" und "hunderte Zellen pro Bild" sauber ab. Gleichzeitig als Metrik mitzählen: Anzahl Komponenten mit mehr als einem Soma.

8. **Erst danach die Filter aus VNR, jeweils angepasst.** (a) Kleinstkomponenten < 5 px entfernen und die entfernte Pixelmenge protokollieren — auf dem WF-Bild sind das 1798 von 3964 Komponenten, das ist eine Entscheidung, keine Aufräumarbeit. (b) Winkelschwellen anziehen, aber **nicht auf 0.7071 raten**: die manuellen DIV10-Tracings künstlich durchtrennen und die tatsächliche Winkelverteilung an echten Unterbrechungen messen, daraus eine Staffelung nach Lückenweite ableiten. Aktuell sind die Defaults extrem permissiv (−0.25 bis −0.65). (c) Bresenham-Veto als *"Weg läuft entlang bestehendem Skelett"* (zusammenhängender belegter Lauf > 3 px oder > 50 % Belegung), **nicht** als "Weg kreuzt Skelett". (d) Gegenseitigkeit der nächsten Nachbarn als Score-Bonus, nicht als hartes Filter. (e) Selbstskalierende Lückengrenze — billig, geringer Effekt, geringes Risiko.

9. **Gerade statt Polynom.** `iterative_postprocessing_stage1_polynomial.py` verwendet Polynomfits; VNR verwirft Splines explizit. Für Lücken unter ~15 px `skimage.draw.line` als 8-zusammenhängende 1-px-Linie zeichnen — dann entfällt jede nachträgliche Skeletonisierung. Für längere Lücken den vorhandenen wahrscheinlichkeitsgeführten Minimalpfad (`route_through_array`, in stage5 Zeile 538 bereits importiert) mit Kosten −log(p_skel), zwingend auf ein Fenster um das Kandidatenpaar beschränkt (`--route-margin 8` existiert).

### Einschränkungen

**Nur der Abstract war lesbar.** Es ist unbekannt, wie Du et al. ihre Schwellwerte bestimmen, welche geometrischen Faktoren ihr Rekonnektionsalgorithmus benutzt, und wie sich der Gewinn auf Netz, Hysterese und Rekonnektion verteilt. Die Konnektivitätsbehauptung wird im Abstract nicht mit einer Konnektivitätsmetrik belegt — genannt werden anscheinend nur Accuracy-Werte, und Accuracy ist bei unter 10 % Vordergrundanteil die aussagearmste denkbare Metrik.

**Die Perzentilregel ist hier gegenstandslos und der vorgeschlagene Weg gefährlich.** 6 % sicherer Vordergrund und 13 % Restfläche stammen aus Fundusbildern, wo Gefäße 8–12 % der Fläche bedecken. Hier: 0.1 %. Der Vorschlag "f aus den GT-Masken messen, dann T_high = quantile(p, 1−f)" ergibt gemessen f = 1.13 % in den Trainingscrops gegen 0.103 % (WF) bzw. 0.19 % (CF) im Übersichtsbild — **Faktor 11**. T_high landete auf dem 98.87-Perzentil, weit unter 0.5, mitten im Rauschen; T_low beim 97.5-Perzentil, bei 86 MP über 2 Millionen Pixel. Das Bild würde geflutet. Die Crops sind eng um je eine Zelle geschnitten; ihr Vordergrundanteil sagt über ein Übersichtsbild nichts aus. Hinzu kommt: die Zelldichte variiert zwischen und innerhalb der Bilder stark — ein globales Bildperzentil ist prinzipiell die falsche Statistik.

**Ein Gefäßbaum gegen hunderte Zellen.** VNR erzwingt genau eine Komponente. Hier gibt es hunderte bis tausende *korrekte* Komponenten. Jede Regel, die zur größten Komponente hin verbindet, ist direkt schädlich. "Rebranch or Remove" mit Löschen der Reste ist unbrauchbar — die Reste sind hier 50–75 % des Signals.

**Das Bresenham-Veto wird bei 1-px-GT strenger als beabsichtigt.** In der Vorlage trifft ein Weg, der ein Gefäß kreuzt, viele belegte Pixel. Bei einem 1-px-Skelett trifft eine echte Kreuzung nur 1–2 Pixel. Ein Veto "irgendein Pixel belegt" verbietet damit genau die korrekten Verbindungen an Kreuzungen — an der Stelle, die im Projekt als Problem gemessen wurde.

**Richtungsbasierte Kriterien scheitern an der Fragmentgröße.** Median-Komponentengröße im argmax: 6 px (WF), 13–27 px nach Hysterese. Bei einem 6-px-Blob gibt es weder Tangente noch Sehne, die eine Richtung definiert. Die Hälfte aller Komponenten ist für jedes Richtungskriterium unsichtbar.

**Die Somas sind auf mindestens einer Bildklasse der Engpass.** WF-Bild: 1534 Soma-Komponenten, Median 2 px, nur 360 mit ≥ 20 px. Der Soma-Kanal ist dort weitgehend zerfallen. Die Spanne "6 bis 60 % Anbindung" ist zu einem erheblichen Teil ein Soma-Problem. Auf CF_30XLP_binningon werden auf 4684×4538 px nur 24 Soma-Instanzen ≥ 200 px gefunden — dort kann der Anbindungsanteil gar nicht steigen, egal welche Skelettschwelle man wählt. Der Pseudo-Dice 0.9357 stammt aus der Validierung auf Einzelzell-Crops und sagt über Übersichtsbilder nichts.

**Die Dickenpropagation entfällt ersatzlos** — sie setzt eine dicke Maske voraus. Ebenso die DVAE-Warnung: Konnektivität durch Verdicken ist hier ein Zielverfehler. Zur Skeletonisierung: sie kostet gemessen 33 % der Pixel (131.990 → 88.000) und 0.7 Prozentpunkte Anbindung (9.3 → 8.6 %). Deshalb Verbindungen von vornherein als 1-px-Linien zeichnen.

**Das Abnahmekriterium "eine Größenordnung weniger Komponenten" ist unerreichbar** und würde die einzige billige wirksame Maßnahme nach dem ersten Test wegwerfen. Ersetze es durch vier gleichzeitig zu erfüllende Bedingungen: Anbindung steigt; Multi-Soma-Komponenten steigen nicht; Komponenten ab 5 px sinken (nicht die rohe Zahl); Zellen, die `quality_gate_cells_for_evo_soma_strict.py` passieren, steigen. Nur die letzte misst den tatsächlichen Lieferwert.

**Es fehlt die Referenz.** Ohne mindestens 3–5 vollständig annotierte Übersichtskacheln (~2000×2000 px) gibt es keine Erfolgsmessung, sondern Selbstbestätigung an Proxy-Zahlen.

**Realistische Nutzenerwartung:** Die Hysterese ist der beste Aufwand-Nutzen-Posten im ganzen Dokument — auf einem Bild 1582 → 565 Komponenten, Median 6 → 44 px, 43.4 → 82.1 % Anbindung, bei null Kosten. Auf anderen Bildern deutlich schwächer (Faktor 2, +3–7 Prozentpunkte) und größtenteils Staubschlucken. Das Rekonnektionsprogramm hat die gemessene Obergrenze 25–53 %; realistisch mit gerichteten Filtern und Fusionsschutz etwa die Hälfte davon, also grob 15–40 % statt 6–24 %. **Die These "Fragmentierung ist ein Binarisierungsproblem" trägt für dieses Projekt nur zu einem kleinen Teil** — bei Median 150 px Abstand zum nächsten Soma liegt die Ursache woanders.

---

## 5. G-Cut: Precise segmentation of densely interweaving neuron clusters

*Li, Zhu, Li, Bienkowski, Foster, Xu, Ard, Bowman, Zhou, Veldman, Yang, Hintiryan, Zhang, Dong. Nature Communications 10:1549, 2019*
Volltext gelesen (Europe PMC XML mit allen Gleichungen) plus vollständiger Quellcode über die Bitbucket-API. Nicht gelesen: die Supplementary Information (über PMC nicht abrufbar) und die Source-Data-Datei mit den numerischen MES-Werten.

### Was das Paper sagt

G-Cut löst **nicht** das Segmentierungsproblem, sondern ausschließlich das nachgelagerte Zuordnungsproblem: Gegeben ein bereits getracter, zusammenhängender Graph aus mehreren verflochtenen Neuronen plus bekannte Soma-Positionen, wird jeder Ast genau einem Soma zugewiesen.

**Graphaufbau.** Vier Knotentypen: Soma-Knoten (extern vorgegeben), Branch-Knoten (mehr als zwei Nachbarn), Leaf-Knoten (genau einer), Path-Knoten (genau zwei). Die ersten drei heißen topologische Knoten. Ein *Ast* ist "a curve immediately connecting two topological nodes", also die maximale Kette von Path-Knoten dazwischen. Die Path-Knoten verschwinden als LP-Variablen, ihre Koordinaten bleiben als Polylinie erhalten. Die SWC-Kantenrichtung wird explizit verworfen — der Graph wird als ungerichtet behandelt.

**Growth Orientation Feature (GOF)** ist der Kern. Ein Ast ist eine nach Bogenlänge parametrisierte Kurve C(l):

```
θ(l) = arccos ⟨ (C(l) − s) / |C(l) − s| , C'(l) ⟩          (Gl. 2)
GOF(C, s) = (1/L) ∫₀^L θ(l) dl                              (Gl. 1)
```

s ist die Soma-Koordinate, C'(l) die Einheits-Tangente. θ ist also der Winkel zwischen "wohin der Ast zeigt" und "radial vom Soma weg". θ = 0 heißt perfekt vom Soma weg (Tropismus), θ = π heißt direkt auf das Soma zu. GOF ist der längengewichtete Mittelwert über [0, π]. **GOF ist richtungsabhängig** — jeder ungerichtete Ast hat pro Soma zwei Kandidatenkosten.

*Implementierungsabweichung:* Der Code (`gcut/neurite.py`) rechnet `np.mean(np.arccos(projection))`, also den **ungewichteten** Mittelwert, und benutzt als Aufpunkt den Knoten p_{j−1} statt der Segmentmitte p_j*. Bei ungleichmäßiger Abtastung — Rasterskelett gegen SWC-Tracing — ist der Unterschied nicht klein.

**Das Prior.** Aus über 70.000 NeuroMorpho.Org-Rekonstruktionen wurde das GOF-Histogramm gebildet, getrennt nach 8 Spezies, 14 Hirnregionen und 2 Zelltypen. Daraus die Tail-Verteilung TailDist(x) = ∫_x^π PDF(ρ)dρ = 1 − CDF(x), also P(GOF ≥ x). Kleiner Winkel → nahe 1 → plausibel; großer Winkel → nahe 0. Im Code als Polynom vom Grad 20 in der standardisierten Variablen z = (θ − π/2)/0.9206291925364665 gespeichert, für alle 24 Schlüssel mit identischem mean und std.

**Kosten:** fitness = weight · TailDist[GOF], penalty g = weight · (1 − TailDist[GOF]). Im Code konkret: `_weight = log(1 + path_length)`; `gof_fitness = P(GOF ≥ θ) · log(1 + 1/branch_order)`; `gof_cost = (1 − fitness) · log(1 + path_length)`. **Der Faktor log(1 + 1/branch_order) steht nicht im Paper.** Bei order 1 ist er 0.693, bei order 5 nur 0.182 — ein starker, undokumentierter Tiefen-Penalty.

**Problemzerlegung.** BFS von Blättern zu Somata: wird nur ein Soma erreicht, ist alles dazwischen eindeutig und fällt aus dem Problem. Dijkstra für jedes Soma-Paar, wobei die Kosten, einen fremden Soma-Knoten zu *verlassen*, auf unendlich gesetzt werden — Wachstumspfade laufen nie durch ein fremdes Soma. Die Äste auf diesen Pfaden ("common path"/"freeway") sind mehrdeutig; abgehende Seitenbäume ("ramps") bekommen je eine einzige LP-Variable mit der Kostensumme des Teilbaums. Das reduziert das LP massiv.

**Das lineare Programm.** w_{C,s} ∈ [0,1] ist der Zugehörigkeitsgrad von Ast C zu Soma s:

```
min Σ_{C,s} w_{C,s} · g_{C,s}
s.t.  Σ_s w_{C,s} = 1                    (Gl. 9)
      w_{C,s} ≥ 0                        (Gl. 10)
      w_{C,s} ≤ w_{par(C,s), s}          (Gl. 11)  ← Topologie
      s* = argmax_s w_{C,s}              (Gl. 12)
```

par(C,s) ist der Elternast im **Dijkstra-Baum von Soma s**, nicht im SWC-Baum. Die Begründung: ein stromabwärts liegender Ast wächst aus einem stromaufwärts liegenden, seine Zugehörigkeit kann also nicht höher sein. Damit kann kein Ast einem Soma zugeordnet werden, ohne dass die gesamte Kette bis zum Soma mitgeht — das verhindert "schwebende" Zuweisungen. Gelöst wird mit OR-Tools GLOP.

**Wichtige Relativierung:** Das ist eine LP-**Relaxation**; w ist kontinuierlich. Das versprochene globale Optimum gilt für das LP, nicht für die ganzzahlige Zuordnung. Gl. (12) ist ein Rundungs-Heuristikschritt ohne Gütegarantie. Das Paper analysiert nicht, wie oft die Lösung fraktional ist.

**Ergebnisse.** Auf simulierten Clustern (Einzelneurone zufällig platziert, nahe Astpaare künstlich verbunden — damit existiert exakte Ground Truth) ist G-Cut über alle Größenstufen und Verflechtungsgrade signifikant besser als NeuroGPS-Tree und TREES-Toolbox (Mann-Whitney-U mit BH-Korrektur, p < 0.01). Auf zwei realen Stapeln (CLARITY, 16 GT-Neurone; Golgi-cox, 45 GT-Neurone) ebenfalls. **Der Fließtext nennt keine absoluten MES-Zahlen**, keine Mediane, keine Effektstärken — nur "höher" plus p-Werte. Laufzeiten und Komplexitätsangaben fehlen vollständig.

### Warum das für dich relevant ist

G-Cut adressiert Problem 3 des Projekts (kreuzende Neuriten einzelnen Somata zuordnen) und nur dieses. Die entscheidende Frage ist, wie groß dieses Problem ist. Gemessen auf `NewImage2` (5040×5056, 236 Soma-Instanzen), argmax-Arbeitspunkt:

- **8.9 %** der Skelettpixel liegen in Komponenten mit ≥ 2 Somata — das ist der gesamte Wirkbereich
- **56.6 %** sind Waisen ohne jedes Soma
- **34.5 %** liegen in Komponenten mit genau einem Soma und sind trivial zugeordnet
- Es gibt im ganzen Bild **17** strittige Komponenten

Nach der Hysterese-Arbeit (Schwelle 0.02) steigt der Wirkbereich auf etwa 34 %, mit 34 Multi-Soma-Komponenten, die größte mit 5 Somata und 583 Skelettpixeln; insgesamt 85 von 236 Somata beteiligt.

**Und es gibt bereits Konkurrenz im Projekt.** In `20280812_newTest2\07_dataset137_paper_guided\07_cell_assignment_comparison\NewImage2` sind drei Methoden vermessen:

| Methode | zugewiesen | mehrdeutig | sichere Zellen |
|---|---|---|---|
| topology_baseline | 79.3 % | 8.4 % | 92 |
| matrix_forest | 84.1 % | 3.6 % | 110 |
| gcut_inspired | **87.3 %** | **0.37 %** | **126** von 156 |

`unassigned_fraction` ist bei allen drei identisch 0.1228 (Komponenten ohne Soma — methodenunabhängig). Ein exaktes G-Cut-LP konkurriert also gegen 87.3 % bei 0.37 % Mehrdeutigkeit. Der Spielraum zwischen bester und schlechtester Methode beträgt 8 Prozentpunkte.

Das vorhandene `gcut_directional_assignment` ist ein Dijkstra über den Zustandsraum (Pixel × Einlaufrichtung) mit Drehstrafe `turn_weight·(1−cos)/2` und Radialstrafe `radial_weight·(1−cos)/2` und führt pro Knoten die zwei besten Somata. Ihm fehlen gegenüber echtem G-Cut drei Dinge: die Kontraktion auf Äste als atomare Einheiten, das empirisch gemessene Prior statt eines linearen (1−cos)/2-Terms, und die Topologie-Nebenbedingung Gl. (11).

**Der Nutzen der letzten lässt sich vorab abschätzen, ohne das LP zu bauen.** Bei soma-*unabhängigen* Kantenkosten erfüllt eine Kürzeste-Wege-Zuordnung Gl. (11) beweisbar automatisch: für den Vorgänger u von v gilt d_t(u) ≥ d_s(u), sobald d_s(v) minimal ist. Hier sind die Kosten nur über die Radialstrafe soma-abhängig, und deren Gewicht ist 0.22 gegenüber 1.0 Schrittkosten. Der Anteil an Knoten, bei denen Gl. (11) überhaupt verletzt sein kann, ist damit klein — und direkt zählbar.

### So wendest du es an

1. **Realitätscheck zuerst.** G-Cut adressiert weder den Zerfall in 1000–12000 Komponenten noch die 6–60 % ohne Soma-Anschluss. Im Gegenteil: die Referenzimplementierung **stürzt ab**, wenn eine Komponente kein Soma enthält (`assert len(unit_soma_ids) > 0` in `gcut.py`). Solange die Konnektivität nicht oben liegt, bringt G-Cut nichts. Reihenfolge: erst Konnektivität, dann Zuordnung.

2. **Wirkbereich messen (halber Tag).** Anteil Skelettpixel in Multi-Soma-Komponenten, Waisenanteil, Komponentenzahl und Mediangröße, DT-Halbbreite. **Entscheidungsregel: bleibt der erste Wert unter etwa 15 %, wird das LP nicht gebaut.**

3. **Billige Vorab-Ablation des LP-Nutzens (halber Tag).** Im bestehenden `gcut_directional_assignment` zählen, bei wie vielen Knoten der Dijkstra-Vorgänger unter dem zugewiesenen Label ein anderes Label trägt. Das ist exakt die Verletzung von Gl. (11) und damit die harte Obergrenze dessen, was das LP ändern kann. **Unter etwa 2 % der Knoten: LP nicht bauen, Vorhaben beenden.**

4. **Ambiguität sofort weich exportieren, unabhängig vom LP (Stunden).** Das vorhandene Verfahren berechnet bereits `node_best_cost`, `node_second_cost` und `relative_margin` und binarisiert sie über `--gcut-margin` auf Status 1/2. Diese Marge stattdessen als kontinuierliches Feld in die QC-Ausgabe und `cell_report.csv` legen. Das hilft der manuellen Kuratierung sofort.

5. **Nur wenn Schritt 2 und 3 dafür sprechen: eigenes GOF-Prior — aber aus den SWC-Tracings, nicht aus den 138er-Crops.** Die Crops sind zu klein: Median 82×97 px, Soma-Median 1020 px (Äquivalenzradius ~18 px), Skelett-Median 98 px auf 2 Komponenten, Ausdehnung ~92 px. Ein Ast reicht typisch von Radius ~18 bis ~46 px — fast nur der proximale Stumpf, der per Konstruktion radial wegzeigt (GOF nahe 0). Ein daraus geschätztes Prior beschreibt genau nicht das, was strittig ist. Richtige Quelle: die originalen `seg.swc` auf FOV-Ebene, wo die Neuriten ungeschnitten sind.

6. **Polylinie auf konstante Bogenlänge resampeln (5–9 px), bevor GOF berechnet wird.** Auf einem 1-px-8-Nachbarschafts-Skelett ist die Tangente auf acht Richtungen quantisiert. GOF über einen 40-px-Ast ist dann ein Mittel über 40 Winkel aus einem Alphabet von 8 Werten — ein Kammspektrum, nicht die glatte Dichte, die das Original mit einem Grad-20-Polynom modelliert.

7. **Verteilung als empirische CDF, nicht als Polynom.** Das Original hat in `distribution.py` ein `assert 0 <= P <= 1`, das bei einem eigenen Fit kippt, weil Polynome hohen Grades am Rand negativ werden. Kumulierte Histogrammsummen plus `np.interp` — monoton, garantiert in [0,1], drei Zeilen.

8. **Trennschärfe mit korrektem Nullmodell messen, bevor das Prior verwendet wird (1 Tag).** Nicht "fremdes Soma aus einem anderen Crop an relativer Position", sondern ein **reales fremdes Soma aus demselben FOV in realistischem Abstand** — nur das ist der Fall, den das Verfahren später entscheiden muss. Kennzahl: AUC des GOF als Klassifikator eigenes-gegen-fremdes Soma. **Unter etwa 0.60: Prior verwerfen** und bei der vorhandenen linearen Radial- plus Drehstrafe bleiben, die ohnehin die linearisierte Fassung desselben Priors ist.

9. **LP nur für die wenigen strittigen Komponenten.** Bei 34 Komponenten mit ≥ 2 Somata ist das ein Spielzeug-LP; `scipy.optimize.linprog(method='highs')` mit `csr_matrix` reicht. Aufbau: Variablenvektor der Länge n_somata × n_äste, bounds (0,1); A_eq mit einer Zeile pro Ast (Einsen über alle Somata), b_eq = 1; A_ub mit einer Zeile pro (Kind, Eltern, Soma), +1 an der Kind-Spalte, −1 an der Eltern-Spalte, b_ub = 0. Deckelung und Kachelung sind bei diesen Zahlen kein Thema. *(Nebenbei: die Behauptung, ortools habe keine Python-3.14-Wheels, ist falsch — geprüft per `pip --dry-run`: ortools 9.15.6755 und skan 0.13.1 mit numba 0.67.0 installieren sauber. scipy ist trotzdem die bessere Wahl, weil es schon da ist.)*

10. **Branch-Order-Faktor nicht übernehmen.** Default aus (Faktor 1). Er steht nicht im Paper, ist eine Codeeigenheit und auf einem Rasterskelett mit vielen Verzweigungsclustern systematisch schädlich, weil die topologische Ordnung dort viel größer ausfällt als bei einem SWC-Tracing. Nur als Schalter führen, um das Original nachstellen zu können. Analog `weight = log(1 + L)` als `--length-weight {log, linear, none}`.

11. **Als vierte Methode in `compare_cell_assignment_methods.py` einhängen** (`gcut_lp_assignment`, neben `gcut_directional_assignment`). Damit erbst du QC-PNG, `cell_report.csv`, `summary.json` und HTML-Übersicht ohne Zusatzarbeit.

### Einschränkungen

**Der Wirkbereich ist gemessen klein.** 8.9 % der Skelettpixel, 17 strittige Komponenten. Der vorgeschlagene Aufbau steht in keinem Verhältnis dazu.

**Es gibt keinen Schiedsrichter.** Es existiert kein Ground Truth, in dem mehrere Zellen desselben Bildes mit Instanz-ID getract sind. Alle Kennzahlen in `compare_cell_assignment_methods.py` (`assigned_fraction`, `ambiguous_fraction`, `safe_cell_count`) sind selbstreferenziell — eine Methode, die alles einem Soma zuschlägt, gewinnt. Der als "die Ablation, die das Paper schuldig bleibt" gefeierte A/B-Vergleich wäre damit nicht entscheidbar. **Das ist der eigentliche Engpass.**

**Harte Voraussetzungen des Verfahrens.** Es braucht einen *zusammenhängenden* Graphen und bekannte Somata. Es repariert keine Konnektivität, fügt keine Kanten hinzu, schließt keine Lücken. Die Autoren sagen selbst, dass Tracing-Fehler des Vorgängers weitergereicht werden und Fehlerkorrektur zukünftige Arbeit ist. Die Python-Fassung liest striktes SWC, also einen Baum — Zyklen sind nicht darstellbar; nur die Matlab-Fassung nimmt eine allgemeine Adjazenzmatrix.

**Morphologie-Annahmen brechen bei Glioblastomzellen.** Der GOF-Prior ist im Kern ein Tropismus-Prior: Neuriten wachsen glatt und vom Soma weg. GBM-Fortsätze sind kürzer, gewundener, oft tangential. Der Prior wurde aus **3D**-Rekonstruktionen geschätzt, dieses Projekt ist 2D — eine Projektion verzerrt die Winkelstatistik systematisch. Die mitgelieferten Verteilungen sind doppelt unpassend. Fällt das GOF-Histogramm auf eigenen Daten flach aus, degeneriert G-Cut zu gewichteter Kürzester-Weg-Zuordnung.

**Die Prämisse "ein Ast gehört genau einem Soma" ist bei Tumor-Mikrotuben teilweise falsch**, nicht bloß schwierig. Bei einer 2D-Kreuzung teilen sich zwei Zellen physisch dieselben Pixel; jede harte Zuweisung ist dort per Konstruktion zur Hälfte falsch. Hinzu kommt: das GT wurde pro Zelle einzeln getract, und wie der Tracer an Kreuzungen entschieden hat, ist nirgends dokumentiert. Man würde gegen eine Referenz optimieren, deren Kreuzungskonvention unbekannt ist.

**Sachfehler zum Längengewicht, den man nicht übernehmen sollte:** "weight = log(1+L) heißt, kurze Fragmente sind fast gewichtslos" stimmt nicht. log(1+10) = 2.40, log(1+40) = 3.71, log(1+600) = 6.40 — das Verhältnis lang zu kurz ist 1.7 bis 2.7, nicht null. Bei Median-Komponentengrößen von 4 bis 44 px dominieren die vielen kurzen Fragmente die Zielfunktion sogar zahlenmäßig.

**X-Junction-Auflösung greift kaum.** Auf einem 1-px-8-Nachbarschafts-Skelett erscheinen echte Kreuzungen fast nie als sauberer Grad-4-Knoten, sondern als zwei benachbarte Grad-3-Knoten oder als kleiner Cluster. Die Regel muss auf "Verzweigungsknoten innerhalb von k px zu einem Cluster zusammenfassen, dann einlaufende Äste paarweise nach Richtungskontinuität koppeln" umgeschrieben werden — mit Tangenten über mindestens 15 Skelettschritte per Total-Least-Squares, sonst dominiert das Treppenmuster.

**Der Rückkanal ins Training ist Selbstbestätigung.** G-Cut-korrigierte Zellen stammen aus derselben Prediction plus einem geometrischen Prior; das Netz würde auf seine eigenen Ausgaben trainiert. Pseudo-Dice stiege, ohne dass die Fragmentierung besser wird. Zusatzbefund: **das GT ist nicht die Fehlerquelle** — in 200 zufällig geprüften Dataset138-Labels sind 100 % der Skelettpixel mit dem Soma verbunden.

**Das ungenannte Hauptproblem.** Das Netz malt ungefähr die richtige *Menge* Skelett, aber als gestrichelte Linie: auf NewImage2 stehen 35.385 Skelettpixel bei 236 Somata (150 px pro Zelle) gegen ein GT-Median von 98 px — aber verteilt auf 1582 Komponenten mit Mediangröße 6 px statt auf 236 zusammenhängende Bäume. Plausible Ursache ist das Trainingsregime: ausschließlich Einzelzell-Crops bei Patch 96×96, das Netz hat nie eine dichte Szene gesehen. **G-Cut berührt davon nichts.**

**Realistische Nutzenerwartung:** Klein bis null. Der Wirkbereich beträgt heute 8.9 %, nach Konnektivitätsarbeit ~34 %, und die Konkurrenz erreicht dort bereits 87.3 % bei 0.37 % Mehrdeutigkeit. Realistisch für LP plus eigenes Prior: eine einstellige Zahl zusätzlich geretteter Zellen, möglicherweise null — nach ein bis zwei Wochen Arbeit. Das mitgelieferte NeuroMorpho-Prior und der Branch-Order-Faktor sind null bis negativ. **G-Cut löst hier ein Problem, das dieses Projekt zu über 90 % nicht hat.**

---

## Was ich zuerst machen würde

**1. Die In-Domain-Topologie-Baseline festschreiben (0.5 Tag, keine Inferenz, keine GPU).**

Über die 283 vorhandenen fold-0-Validierungsvorhersagen: β0^A(Klasse 1 ∪ Klasse 2), Anteil Fälle mit β0 = 1, Anteil somaverbundener Skelettpixel, Skelettpixelzahl, Dice Klasse 1 — pro Fall, nie über Dimensionen aggregiert. Die Zahlen liegen bereits vor und lauten 5.60 / 44.9 % / 73.0 % / 245 px / 0.719 gegen GT 1.00 / 100 % / 100 % / 205 px.

*Begründung:* Das ist die wichtigste Zahl im ganzen Dokument und sie fehlte bisher. **Das Modell versagt topologisch bereits auf sauberen Einzelzell-Crops, in perfekter Domäne, mit genau einer Zelle im Bild.** Nur 44.9 % der Validierungsfälle sind eine einzige Komponente, wo das GT zu 100 % eine ist. Die 6–60 % auf Übersichtsbildern sind also nicht primär ein Domänen- oder Nachbearbeitungsproblem, sondern die Fortsetzung eines Trainingsdefizits. Wer daraufhin weiter an der Nachbearbeitung optimiert, optimiert am kleineren Teil des Problems. Ohne diese Tabelle ist außerdem kein einziges Folgeexperiment als Fortschritt belegbar.

**2. Den Skalentest fahren (0.5 Tag, nur Neuinferenz, kein Training).**

`nnUNetPlans.json` enthält `original_median_spacing_after_transp [999.0, 1.0, 1.0]`, die Trainings-TIFFs tragen XResolution 1.0 — **nnU-Net normiert die Skala nie**. Die DIV10-Tracings entstanden bei 0.4062 µm/px, `NewImage2_0000.tif` trägt 0.2875 µm/px. Ein Übersichtsbild mit den Faktoren 0.5, 0.7, 1.0, 1.41, 2.0 resampeln (bikubisch), jeweils vorhersagen, das Label zurückskalieren, mit den Metriken aus Schritt 1 bewerten. Kosten: ~2.5 min Inferenz pro Skala auf 25 MP.

*Begründung:* Ein unkorrigierter Skalenfaktor von 41 % ist bei einer 1–2 px breiten Zielstruktur ein Fehler **erster Ordnung** — bei den 40x-Bildern ist er noch größer. Der Objektivtest zeigt genau dieses Muster empirisch ("Binning ist der schärfste Trenner", IoU Skelett nur 0.37–0.59 zwischen Netzen). Das ist nie getestet worden, kostet einen halben Tag, braucht kein Training, und wenn hier ein Optimum außerhalb von 1.0 liegt, ist ein Großteil des Problems ohne jede Modelländerung gelöst. Die Dauerlösung wäre dann, allen Bildern korrekte Spacing-Sidecars zu geben, damit nnU-Net selbst resampelt. Direkt anschließen: derselbe Test für die Normalisierung — dasselbe Bild einmal als Ganzes und einmal gekachelt vorhersagen, weil nnU-Net den Z-Score pro Eingabedatei rechnet und der Kontrast zwischen Training (1.13 σ) und Übersicht (0.46 σ) um Faktor 2 auseinanderläuft.

**3. Den Hysterese-Sweep sauber über alle vorhandenen `.npz` laufen lassen (0.5–1 Tag, keine Inferenz).**

Seed p ≥ 0.5, Wachstum über Zusammenhangskomponenten bis 0.30 / 0.20 / 0.10 / 0.05 / 0.02. Somamaske unverändert aus dem argmax übernehmen. Pro Schwelle protokollieren: Maskenpixel, Pixel nach `skeletonize`, Komponentenzahl gesamt **und ab 5 px**, größte und mediane Komponente, Anteil am Soma, DT-Halbbreite, **und den Medianabstand der neu hinzugekommenen Pixel zum nächsten Soma**. Nicht neu implementieren — `recover_skeleton_from_probabilities` in `postprocess_overview_v3.py` (ab Zeile 243) portieren.

*Begründung:* Das ist die einzige Maßnahme im ganzen Dokument, die bei null Kosten einen substanziellen gemessenen Gewinn gezeigt hat: auf NewImage2 1582 → 565 Komponenten, Median 6 → 44 px, 43.4 → 82.1 % Soma-Anbindung, und die Kontrollrechnung bestätigt echtes longitudinales Lückenschließen (neue Pixel im Median 48.8 px vom Soma entfernt). Auf anderen Bildern ist der Gewinn deutlich kleiner und größtenteils Staubschlucken — genau deshalb muss der Sweep über *alle* vorhandenen Karten laufen und die Statistik "ab 5 px" mitführen, sonst verkauft man sich einen Erfolg, den es nicht gibt. Es gibt hier auch einen Widerspruch in den Vormessungen: ein globaler Schwellwert-Sweep verschlechterte die Komponentenzahl, das komponentenbasierte Hysterese-Wachstum verbesserte sie. Der Unterschied ist die Anbindungsbedingung, und ihn sauber zu vermessen ist die Aufgabe.

**Was direkt danach kommt, aber nicht in dieselbe Woche gehört:** Der teuerste und unverzichtbare Posten ist ein Instanz-Ground-Truth auf Übersichtsebene — 3 bis 8 vollständig getracte Kacheln von etwa 2000×2000 px, alle Zellen, nicht nur eine kuratierte, permanent aus dem Training draußen. 2–5 Tage überwiegend manuelle Arbeit. Ohne diese Kacheln lässt sich bei **keiner** der übrigen Maßnahmen unterscheiden, ob mehr Skelettpixel echte Struktur oder Falschpositive sind. Der aktuelle Trainer liefert gegenüber der Baseline +84 % Skelettpixel und hebt "Skelett am Soma" von 9.3 % auf 49.3 % — und genau diese Zahlen könnten sowohl ein Erfolg als auch die von arXiv:2508.11374 beschriebene FP-Inflation sein.