# Priorisierte Leseliste — Glioblastom-Neuritensegmentierung mit nnU-Net v2

[Zur Dokumentationsübersicht](../README.md)

Alle Zitationen wurden gegen den jeweiligen Prüfbericht abgeglichen und, wo eine Korrektur vorlag, in der korrigierten Fassung übernommen. Kein Eintrag war mit `exists=false` markiert; entfernt wurden ausschließlich Dubletten über Themenfelder hinweg (NeuronCyto II, SNT, Carneiro-Esteves Neurocomputing, Quality Control of Neuron Reconstruction, clDice). Jedes dieser Papers steht nun einmal an der Stelle, wo es den größten Hebel hat, mit Querverweis von der zweiten Stelle.

---

## Wenn du nur 5 Paper liest

**1. Berger, Lux, Weers, Menten, Rueckert, Paetzold: *Pitfalls of Topology-Aware Image Segmentation.* IPMI 2025, Springer, S. 297–312.**
DOI: https://doi.org/10.1007/978-3-031-96628-6_20 · Preprint: arXiv:2412.14619 · Code: https://github.com/AlexanderHBerger/topo-pitfalls

Zuerst lesen, weil es die Möglichkeit eröffnet, dass ein Teil deines Hauptproblems ein Messartefakt ist. Ein diagonal verlaufendes 1-Pixel-Skelett ist unter 4-Konnektivität zerrissen und unter 8-Konnektivität durchgehend — `scipy.ndimage.label()` verwendet in 2D standardmäßig die 4-konnexe Struktur. Bevor du irgendetwas am Training änderst, zähle deine „hunderte bis tausende Komponenten" und den „Anteil somaverbundener Skelettpixel" mit `structure=np.ones((3,3))` neu. Das kostet zehn Minuten und kann die Problemgröße halbieren.

**2. Öner, Kozinski, Citraro, Fua: *Adjusting the Ground Truth Annotations for Connectivity-Based Learning to Delineate.* IEEE TMI 41(12):3675–3685, 2022.**
DOI: https://doi.org/10.1109/TMI.2022.3193072 · Preprint: arXiv:2112.02781 · Code: https://github.com/doruk-oner/AdjustingAnnotationswithSnakes

Bei einer 1-Pixel-Klasse ist ein Annotationsversatz von einem Pixel bereits 100 % Fehler — das Netz wird gleichzeitig für einen False Positive und einen False Negative bestraft und lernt daraus, unsichere, ausgedünnte Skelette zu produzieren. Das ist exakt dein Symptombild und ein Mechanismus, den keine Loss-Änderung repariert, solange die Labels so behandelt werden. Bei 1400 Crops mit Median 90×89 px ist die vorgeschlagene Annotation-Refinement-Schleife rechnerisch trivial und potenziell der größte Einzelhebel.

**3. Kirchhoff, Rokuss, Roy, Kovacs, Ulrich, Wald, Zenk, Vollmuth, Kleesiek, Isensee, Maier-Hein: *Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures.* ECCV 2024, LNCS, S. 218–234.**
DOI: https://doi.org/10.1007/978-3-031-72980-5_13 · Preprint: arXiv:2404.03010 · Code: https://github.com/MIC-DKFZ/Skeleton-Recall

Du benutzt bereits eine Variante davon, kennst aber möglicherweise nicht die zwei Designentscheidungen, die bei 1-px-GT alles entscheiden. Erstens der Tube-Radius: der Loss dilatiert das GT-Skelett zu einem Schlauch, und dessen Radius bestimmt, ob 1–2 px Versatz toleriert werden. Zweitens ist es ein reiner **Recall**-Term ohne False-Positive-Bestrafung — deine 2×-Gewichtung darauf begünstigt ausgefranste, schwach-konfidente Vorhersagen. Reproduziere zuerst die Originalgewichtung mit Dice+CE als Baseline.

**4. Du, Zhang, Song, Bao, Zhang, Wu, Liu: *Retinal blood vessel segmentation by using the MS-LSDNet network and geometric skeleton reconnection method.* Computers in Biology and Medicine 153:106416, 2023.**
DOI: https://doi.org/10.1016/j.compbiomed.2022.106416

Der billigste denkbare Hebel gegen die Fragmentierung, ohne jedes Retraining: die Hysterese-Schwellwertbildung. Ein Großteil deiner Komponenten entsteht vermutlich gar nicht im Netz, sondern erst beim `argmax` über die Softmax — die 1-px-Skelettklasse ist gegenüber Soma und Hintergrund massiv unterrepräsentiert und verliert an jedem schwachen Pixel. Ersetze argmax durch Region-Growing von hohen Seed-Schwellen (z. B. 0.7) mit niedrigem Fortsetzungsschwellwert (z. B. 0.25). Das ist ein Nachmittag Arbeit und misst sofort, wieviel Fragmentierung überhaupt echt ist.

**5. Li, Zhu, Li, Bienkowski, Foster, Xu, Ard, Bowman, Zhou, Veldman, Yang, Hintiryan, Zhang, Dong: *Precise segmentation of densely interweaving neuron clusters using G-Cut.* Nature Communications 10:1549, 2019.**
DOI: https://doi.org/10.1038/s41467-019-09515-0 · Code: https://bitbucket.org/muyezhu/gcut

Das einzige Paper, das die Neurit-zu-Soma-Zuordnung als globales Optimierungsproblem statt als lokale Heuristik löst. Entscheidend ist die Topologie-Nebenbedingung: die Zugehörigkeit eines Astes zu einem Wurzelsoma kann nicht höher sein als die seines unmittelbaren Vorgängerastes — das verhindert genau den Fehler, dass ein distales Fragment einem anderen Soma zugeschlagen wird als sein Elternast. Das Growth-Orientation-Prior musst du aus eigenen kuratierten SWC-Traces neu schätzen, weil Glioblastomzellen eine andere Winkelstatistik haben als kortikale Neurone.

---

## 1. Preprocessing und Domain Shift

### 1.1 Aaron, Chew: *A guide to accurate reporting in digital image processing — can anyone reproduce your quantitative analysis?* Journal of Cell Science 134(6):jcs254151, 2021.
DOI: https://doi.org/10.1242/jcs.254151 · Priorität: hoch

Zeigt, dass allein die Wahl der Hintergrundschätzmethode und ihrer Parameter bis zu ca. 75 % Unterschied in der Gesamtintensität desselben Bildes erzeugt. Damit ist Rollingball kein neutraler Schritt, sondern eine massive radiusabhängige Intensitätstransformation — bei 1 px breiten Neuriten rechnet ein zu kleiner Radius das Signal selbst als Hintergrund weg. Behandle den Radius als Hyperparameter mit eigenem Sweep (5/15/25/50/100 px, getrennt für Widefield und Konfokal) und logge ihn pro Datensatz mit.
*Nicht verwechseln mit dem Schwesterartikel im selben Heft (jcs254144, „…digital image ACQUISITION").*

### 1.2 Peng, Thorn, Schroeder, Wang, Theis, Marr, Navab: *A BaSiC tool for background and shading correction of optical microscopy images.* Nature Communications 8:14836, 2017.
DOI: https://doi.org/10.1038/ncomms14836 · Code: https://github.com/marrlab/BaSiC · BaSiCPy: https://pypi.org/project/BaSiCPy/ · Priorität: hoch (hochgestuft)

Bei 5000×5000 bis 35000×35000 px ist Shading die wahrscheinlichste Erklärung dafür, dass dasselbe Modell in der Bildmitte sauber und am Kachelrand fragmentiert segmentiert. BaSiC schätzt Flatfield und Darkfield ohne manuelle Parameterwahl — bei deiner Vielfalt an Objektiven praktisch entscheidend. **Die Diagnose ist hier wichtiger als die Korrektur:** plotte den Anteil somaverbundener Skelettpixel als Funktion der Position im Übersichtsbild. Fällt er zum Rand hin ab, ist Shading nachweisbar mitschuldig, und BaSiC ist der billigste Hebel. Multiplikativ, gehört also **vor** jeden additiven Hintergrundabzug.

### 1.3 Mahbod, Schaefer, Löw, Dorffner, Ecker, Ellinger: *Investigating the Impact of the Bit Depth of Fluorescence-Stained Images on the Performance of Deep Learning-Based Nuclei Instance Segmentation.* Diagnostics 11(6):967, 2021.
DOI: https://doi.org/10.3390/diagnostics11060967 · Code: https://github.com/masih4/BitDepth_NucSeg · Priorität: hoch

Beantwortet deine uint8-vs-uint16-Frage empirisch: die Ergebnisse sind praktisch identisch (Dice 88.7 vs. 89.2 %), das Netz nutzt Morphologie, nicht exakte Intensitätswerte. Der für dich wichtigere Befund ist der zweite: ohne Normalisierung konvergierte das Training überhaupt nicht, und Perzentil-Clipping mit Min/Max war die beste von vier Varianten. Bittiefe ist also keine eigene Domäne — aber die Normalisierung muss robust sein (z. B. 0.1–99.8 %), nicht naive Division durch 255/65535.

### 1.4 Zhang, Wang, Yang, Roth, Myronenko, Xu, Xu, Sanford, Turkbey, Wood, Harmon: *Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep Stacked Transformation.* IEEE TMI 39(7):2531–2540, 2020. (BigAug)
DOI: https://doi.org/10.1109/TMI.2020.2973595 · Priorität: hoch

Die methodisch sauberste Antwort auf deinen Domain Shift, weil die Situation exakt passt: eine Quelldomäne, mehrere unbekannte Zieldomänen, keine Zieldaten. Neun gestapelte Transformationen erreichen 80.0 % Dice auf ungesehenen Domänen gegenüber 49.8 % Baseline und 63.5 % für CycleGAN-Adaption — ein gutes Argument, teure Domain-Translation gar nicht erst zu bauen. Die Skalierung 0.4–1.6× simuliert direkt den Objektivwechsel. **Achtung bei 1-px-Skelett:** Blur- und Downsample-Augmentation dürfen nur auf dem Bild wirken, nie auf der Maske.

### 1.5 Hüpfel, Kobitski, Zhang, Nienhaus: *Wavelet-based background and noise subtraction for fluorescence microscopy images.* Biomedical Optics Express 12(2):969–980, 2021.
DOI: https://doi.org/10.1364/BOE.413181 · Code: https://github.com/NienhausLabKIT/HuepfelM · Priorität: hoch

Die quantitativ belegte Alternative zum Rolling-Ball. Gegen Ground Truth erreicht WBS niedrigeren MSE (29.6 vs. 34.7) und höheren Pearson-Koeffizienten (0.57 vs. 0.50) als RBA — und, für dich zentral, den Strukturerhalt-Nachweis an dünnen Objekten: die scheinbare Beadgröße verbessert sich auf 440 ± 80 nm gegenüber 530 ± 80 nm. Der Wavelet-Level ist der einzige echte Parameter und lässt sich direkt an die Neuritenbreite koppeln, was reproduzierbarer ist als ein Rollingball-Radius.

### 1.6 Stringer, Pachitariu: *Cellpose3: one-click image restoration for improved cellular segmentation.* Nature Methods 22(3):592–599, 2025.
DOI: https://doi.org/10.1038/s41592-025-02595-5 · Preprint: bioRxiv 2024.02.10.579780 · Code: https://github.com/MouseLand/cellpose · Priorität: hoch

Der konzeptionelle Kernbefund: Modelle, die auf Pixel-Wiederherstellung optimiert sind, sind nicht dasselbe wie Modelle, die auf Segmentierbarkeit optimiert sind. Ein generisches Denoising kann deine Metrik verschlechtern, obwohl die Bilder schöner aussehen — genau das Muster, das du beim Rollingball schon beobachtest. Praktisch: Cellpose3-Restaurationsmodelle als Preprocessing testen und gegen Skelett-Konnektivität messen, nicht gegen PSNR. Ein reiner MSE-Denoiser ist bei 1-px-Strukturen strukturell im Nachteil, weil deren Fehler im Loss verschwindet.

### 1.7 Laine, Jacquemet, Krull: *Imaging in focus: An introduction to denoising bioimages in the era of deep learning.* The International Journal of Biochemistry & Cell Biology 140:106077, 2021.
DOI: https://doi.org/10.1016/j.biocel.2021.106077 · Priorität: hoch

Formuliert die Trennlinie, die deine Pipeline braucht: denoisierte Bilder für Segmentierung und Objektlokalisation ja, intensitätsbasierte Quantifizierung nein — dafür zurück auf die Rohdaten. Für dich heißt das zwei Bildkanäle durch die Pipeline: das verarbeitete Bild geht in nnU-Net, die resultierende Maske wird für alle Intensitätsmerkmale auf das **rohe** Bild zurückprojiziert. Die Warnung vor halluzinierten Strukturen aus impliziten Trainings-Priors ist bei dir besonders heimtückisch: eine halluzinierte Verbindung zwischen kreuzenden Neuriten würde als „endlich zusammenhängendes Skelett" fehlinterpretiert.

### 1.8 Singh, Bray, Jones, Carpenter: *Pipeline for illumination correction of images for high-throughput microscopy.* Journal of Microscopy 256(3):231–236, 2014.
DOI: https://doi.org/10.1111/jmi.12178 · CellProfiler-Module: https://cellprofiler.org · Priorität: mittel

Eines der wenigen älteren Papers, das den **nachgelagerten** Effekt einer Korrektur misst statt nur die Korrektur vorzuschlagen. Übernimm die Evaluationslogik, nicht nur die Methode: berichte jede Preprocessing-Variante gegen eine projektspezifische Downstream-Metrik (Anteil somaverbundener Skelettpixel, Komponentenzahl pro Zelle) statt gegen Dice — ein Dice-neutraler Schritt kann die Konnektivität halbieren. Zweitens: Korrekturfunktion aus vielen Bildern derselben Konfiguration schätzen und als Datei versionieren, sonst führst du selbst eine neue Varianzquelle ein.

### 1.9 Krull, Buchholz, Jug: *Noise2Void — Learning Denoising From Single Noisy Images.* IEEE/CVF CVPR 2019, S. 2129–2137.
https://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf · arXiv:1811.10980 · Priorität: mittel

Das praktisch einzig gangbare Denoising für dich, weil du weder saubere GT-Bilder noch Noisy-Noisy-Paare hast — trainierbar pro Domäne aus den unannotierten Übersichtsbildern. Die Einschränkung folgt direkt aus dem Blind-Spot-Design und trifft dich hart: N2V rekonstruiert jeden Pixel **nur** aus seiner Nachbarschaft, und 1 px breite Neuriten sind genau der Fall, in dem das Signal kostet. Konkreter Test: GT-Skelett über das denoisierte Bild legen und den Kontrast entlang der Neuriten vorher/nachher messen. Bei strukturiertem Rauschen (Zeilenrauschen durch Binning, sCMOS-Fixed-Pattern) ist die Unabhängigkeitsannahme verletzt — dann braucht es die strukturierte Blind-Spot-Variante.

### 1.10 Weigert, Schmidt, Boothe et al. (Myers, Jug): *Content-aware image restoration: pushing the limits of fluorescence microscopy.* Nature Methods 15(12):1090–1097, 2018.
DOI: https://doi.org/10.1038/s41592-018-0216-7 · CSBDeep (Python/Fiji/KNIME) · Priorität: mittel

Nur relevant, wenn du am Mikroskop noch Daten erheben kannst: dann Paare aus (kurze Belichtung / hohes Binning) und (lange Belichtung / kein Binning) aufnehmen und ein CARE-Modell trainieren, das deine schlechten Bedingungen auf die guten abbildet — echte gerichtete Domain-Harmonisierung statt blinder Normalisierung. CARE löst explizit tubuläre Strukturen auf, also deinen Neuritenfall. Ohne neue Aufnahmen bleibt der bescheidenere Punkt: die Trainingsdatenpaarung als Blaupause für ein Degradationsmodell (gute Bilder künstlich verschlechtern und als zusätzliche nnU-Net-Trainingsdaten einspeisen). Zusammen mit Laine et al. (1.7) lesen — CARE ist das Paper, dessen Erfolg die Halluzinationsdebatte ausgelöst hat.

---

## 2. Training: Loss, Labels, nnU-Net-Konfiguration

### 2.1 Berger, Lux, Weers, Menten, Rueckert, Paetzold: *Pitfalls of Topology-Aware Image Segmentation.* IPMI 2025, S. 297–312.
→ siehe Top-5, Platz 1. **Vor allen anderen Trainingsänderungen lesen.**

### 2.2 Kirchhoff et al.: *Skeleton Recall Loss…* ECCV 2024, S. 218–234.
→ siehe Top-5, Platz 3.

### 2.3 Öner, Kozinski, Citraro, Fua: *Adjusting the Ground Truth Annotations…* IEEE TMI 41(12):3675–3685, 2022.
→ siehe Top-5, Platz 2. Praktische Minimalvariante, falls die volle Snake-Schleife zu aufwendig ist: ein Toleranzband im Loss (GT-Skelett dilatiert als `ignore`-Zone um den 1-px-Kern), damit ein 1-px-Versatz nicht doppelt als FP **und** FN bestraft wird.

### 2.4 Shit, Paetzold, Sekuboyina, Ezhov, Unger, Zhylka, Pluim, Bauer, Menze: *clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation.* CVPR 2021, S. 16560–16569.
https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html · arXiv:2003.07311 · Code: https://github.com/jocpae/clDice · Priorität: hoch

Ursprung der gesamten Centerline-Loss-Familie, garantiert Topologieerhaltung bis auf Homotopieäquivalenz. **Der entscheidende Fallstrick für dich:** clDice ist für Masken *mit Dicke* gedacht, aus denen erst ein Skelett extrahiert wird. Bei bereits 1 px breiter GT degeneriert soft-clDice weitgehend zu normalem Dice — das erklärt möglicherweise, warum reine Skelett-Losses deine Fragmentierung nicht beheben. Die Lösung: einen zusätzlichen binären Kanal **„Zelle = Skelett ∪ Soma"** definieren und Topologie-Terme dort rechnen. Dieser Kanal kodiert „Neurit hängt am Soma" als eine einzige zusammenhängende Komponente — also genau deine 20–60-%-Kennzahl als Trainingsziel. clDice ist außerdem primär eine **Metrik**; siehe Abschnitt 5.2.

### 2.5 Acebes, Moustafa, Camara, Galdran: *The Centerline-Cross Entropy Loss for Vessel-Like Structure Segmentation: Better Topology Consistency Without Sacrificing Accuracy.* MICCAI 2024, LNCS 15008, S. 710–720.
DOI: https://doi.org/10.1007/978-3-031-72111-3_67 · Code: https://github.com/cesaracebes/centerline_CE · Priorität: hoch

Direkte Antwort auf zwei clDice-Schwächen, die dich beide treffen: clDice kostet Segmentierungsgenauigkeit und ist **nicht robust gegen verrauschte Annotationen**. Deine 1400 manuell kuratierten 1-px-Skelette sind per Konstruktion verrauscht (Versatz, fehlende dünne Äste, uneinheitliche Kuratierungstiefe). clCE kombiniert die Rauschrobustheit von Cross-Entropy mit dem Topologiefokus von centerline-Dice. Günstigste Einzelmaßnahme gegen die Kombination „Label-Rauschen plus extremes Klassenungleichgewicht", und als kompakte framework-agnostische Funktion direkt in den nnU-Net-v2-Trainer einsetzbar.

### 2.6 Szustakowski, Frank, Esser, Gründemann, Piraud: *Preserving instance continuity and length in segmentation through connectivity-aware loss computation.* arXiv:2509.03154, 2025 (Preprint, kein Peer-Review-Venue).
https://arxiv.org/abs/2509.03154 · Priorität: hoch

Anwendungsseitig das nächstliegende Paper: Fluoreszenzmikroskopie, langgestreckte neuronale Strukturen, Signalabbrüche als Ursache von Diskontinuitäten — und die nachgelagerte Messgröße ist die **Länge**, nicht Dice. Das ist exakt deine Situation mit SWC-Traces. Übernimm erstens den Negative Centerline Loss, der gezielt Hintergrundvorhersagen auf der GT-Mittellinie bestraft — der direkteste Angriff auf Lücken bei 1-px-Skeletten. Zweitens, methodisch mindestens so wichtig: stelle die Zielgröße auf „Anzahl Diskontinuitäten pro Zelle" und „gemessene vs. GT-Neuritenlänge" um.

### 2.7 Lux, Berger, Weers, Stucki, Rueckert, Bauer, Paetzold: *Topograph: An Efficient Graph-Based Framework for Strictly Topology Preserving Image Segmentation.* ICLR 2025 (Spotlight).
https://openreview.net/forum?id=Q0zmmNNePz · arXiv:2411.03228 · Priorität: hoch

Baut einen Component Graph, der die Zusammenhangskomponenten von Prediction und GT vollständig kodiert, und aggregiert den Loss lokal an den topologisch kritischen Pixeln — also genau dort, wo eine Fehlklassifikation eine Komponente zerreißt. Andere Losses bestrafen Lücken nur indirekt, Topograph bestraft das verursachende Pixel. 3–6× schneller als persistente Homologie, bei deinen Bildgrößen relevant. Praktisch: als Zusatzterm auf dem Union-Kanal (Skelett ∪ Soma) mit kleinem Gewicht nach ca. 100 Epochen Warm-up zuschalten. Die mitgelieferte strikte Topologie-Metrik ist zusätzlich direkt als QC-Kennzahl pro Zelle verwendbar.

### 2.8 Berger, Lux, Stucki, Bürgin, Shit, Banaszak, Rueckert, Bauer, Paetzold: *Topologically Faithful Multi-class Segmentation in Medical Images.* MICCAI 2024, LNCS 15008, S. 721–731.
DOI: https://doi.org/10.1007/978-3-031-72111-3_68 · Preprint: arXiv:2403.11001 · Code: https://github.com/AlexanderHBerger/multiclass-BettiMatching · Priorität: mittel

Du hast ein echtes Mehrklassenproblem, bei dem die topologisch relevante Struktur die **Vereinigung** zweier Vordergrundklassen ist: eine Zelle ist genau dann korrekt, wenn Skelett und Soma eine gemeinsame Komponente bilden. Das Paper zerlegt N-Klassen-Segmentierung in N Ein-Klassen-Aufgaben und validiert unter anderem auf Arterie-Vene-Segmentierung — strukturell dasselbe Problem: zwei Vordergrundklassen eines zusammenhängenden Baums. Als günstigere Alternative mit demselben Trick lässt sich Topograph (2.7) auf dem Union-Kanal einsetzen.

### 2.9 Isensee, Wald, Ulrich, Baumgartner, Roy, Maier-Hein, Jäger: *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation.* MICCAI 2024, Springer.
DOI: https://doi.org/10.1007/978-3-031-72114-4_47 · arXiv:2404.09556 · Code: https://github.com/MIC-DKFZ/nnUNet · Priorität: mittel (aber der Patch-Size-Punkt ist hoch)

Zwei Konsequenzen. Erstens: wechsle auf ein ResEnc-Preset (mindestens ResEncM), CNN-Varianten schlagen Transformer und Mamba, und der Hebel liegt in der Skalierung auf modernes GPU-Budget. Zweitens, und das ist der eigentliche projektspezifische Punkt: **löse den Patch-Size-Mismatch auf.** nnU-Net leitet die Patchgröße aus der Medianbildgröße ab — bei Median-Crops von 90×89 px landest du bei winzigen Patches, während die Sliding-Window-Inferenz auf 35000×35000 läuft. Ein Netz, das nur 90×89-Ausschnitte sieht, kann Neuritenverläufe über längere Distanzen nicht kontextuell schließen. Entweder Crops mit Originalkontext neu ausschneiden oder die plans-Datei manuell auf deutlich größere Patches setzen. Nebeneffekt: mehr Kontext hilft auch bei kreuzenden Neuriten.

### 2.10 Park, Lee, Kim, Cecen, Kwon, Jeong: *Neuron Segmentation using Incomplete and Noisy Labels via Adaptive Learning with Structure Priors.* IEEE ISBI 2021, S. 1466–1470.
DOI: https://doi.org/10.1109/ISBI48211.2021.9434102 · Priorität: mittel (unterschätzt — praktisch sofort umsetzbar)

Adressiert exakt die Kombination in deinen Einzelzell-Crops: unvollständige **und** ungenaue Annotationen. In einem 90×89-Crop ragen fast zwangsläufig Neuriten benachbarter Zellen hinein, die nicht annotiert sind — sie werden als Hintergrund gelernt und trainieren das Netz systematisch darauf, dünne Strukturen zu unterdrücken. Ein plausibler Mechanismus für dein fragmentiertes Skelett. **Sofortmaßnahme:** ein `ignore`-Label einführen für alles im Crop, was Signal trägt aber nicht zur Zielzelle gehört. nnU-Net v2 unterstützt Ignore-Labels nativ.

---

## 3. Postprocessing: Rekonnektion und Tracing

### 3.1 Du et al.: *MS-LSDNet + geometric skeleton reconnection.* Comput Biol Med 153:106416, 2023.
→ siehe Top-5, Platz 4. Nach der Hysterese die geometrische Rekonnektion: Endpunkte extrahieren, Tangentenrichtung aus den letzten k Pixeln schätzen, Paare mit kompatibler Richtung und Distanz verbinden. Laufzeit ca. 0,013 s pro Bild — bei deinen Bildgrößen ein Argument.

### 3.2 Carneiro-Esteves, Vacavant, Merveille: *A plug-and-play framework for curvilinear structure segmentation based on a learned reconnecting regularization.* Neurocomputing 599:128055, 2024.
DOI: https://doi.org/10.1016/j.neucom.2024.128055 · Preprint: arXiv:2408.12943 · Code: https://github.com/creatis-myriad/plug-and-play-reco-regularization · Priorität: hoch

Der Rekonnektionsoperator leistet beides, was du bei tausenden Komponenten brauchst: er schließt Lücken zwischen nahen, gleich orientierten Fragmenten **und** verwirft kleine Komponenten, die nur Rauschen sind. Punkt zwei adressierst du bisher gar nicht, ist aber bei „größte Komponente oft nur wenige hundert Pixel" mindestens so wichtig. Der Operator wird unsupervised aus synthetisch zerrissenen Strukturen gelernt — keine neuen Handannotationen nötig, du erzeugst die Trainingspaare aus deinen 1400 GT-Crops mit der beobachteten Lückenlängenverteilung. Berichtet werden ca. 90 % Konnektivitätsgewinn in 2D, mit Transfer auf Straßenrisse und Hornhautzellen (spricht für Domain-Shift-Robustheit). **Reihenfolge beachten: erst rekonnektieren, dann Baumzuordnung** — sonst setzt jede Graph-Methode auf einem zerrissenen Skelett auf.

### 3.3 Carneiro-Esteves, Vacavant, Merveille: *Restoring Connectivity in Vascular Segmentations Using a Learned Post-processing Model.* In: Chen, Singh, Hu (eds), Topology- and Graph-Informed Imaging Informatics (TGI3 2024, MICCAI-Workshop), LNCS 15239, S. 55–65. Springer, 2025.
DOI: https://doi.org/10.1007/978-3-031-73967-5_6 · Preprint: arXiv:2404.10506 · Priorität: hoch

Das Schwesterpaper zu 3.2, mit dem konkreteren Bauplan für eine zweite Netzstufe: Input = vorhergesagte Skelett-Wahrscheinlichkeitskarte (optional plus Rohbild als zweiter Kanal), Target = zusammenhängendes GT-Skelett, anwendbar auf den Output *jedes* Segmentierers. Trainingspaare erzeugst du kostenlos, indem du in den 1400 GT-Crops zufällig Segmente von 3–20 px Länge auslöschst. Läuft danach kachelweise über die Großbilder, **ohne nnU-Net neu zu trainieren.**

### 3.4 Xiao, Peng: *APP2: automatic tracing of 3D neuron morphology based on hierarchical pruning of a gray-weighted image distance-tree.* Bioinformatics 29(11):1448–1454, 2013.
DOI: https://doi.org/10.1093/bioinformatics/btt170 · Code: https://github.com/Vaa3D/vaa3d_tools · Priorität: hoch

Löst „nur 20–60 % der Skelettpixel hängen mit einem Soma zusammen" an der Wurzel, indem die binäre Maske gar nicht erst als Wahrheit genommen wird: der Baum ist per Konstruktion zusammenhängend und wurzelt in einem Startpunkt — Fragmentierung kann nicht auftreten, sie wird durch anschließendes Pruning ersetzt. Für dich der entscheidende Kniff: **nutze die nnU-Net-Skelettwahrscheinlichkeit als Kostenbild** statt der Rohintensität, Fast Marching vom Schwerpunkt jedes Somas über c = 1/(p_skelett + ε). Jedes Fragment wird über den plausibelsten Pfad an genau ein Soma angebunden, und die Neurit-zu-Soma-Zuordnung fällt als Nebenprodukt an. Ausgabe ist direkt SWC.

### 3.5 Liu, Zhang, Song, Peng, Cai: *Automated 3-D Neuron Tracing With Precise Branch Erasing and Confidence Controlled Back Tracking.* IEEE TMI 37(11):2441–2452, 2018. (Algorithmus: Rivulet2)
DOI: https://doi.org/10.1109/TMI.2018.2833420 · Code: https://github.com/RivuletStudio/rivuletpy · Priorität: hoch

Zwei Mechanismen zielen direkt auf deine Probleme. Erstens stoppt die Tracing-Iteration nicht am Rand eines bereits getracten Bereichs, sondern sucht in jedem Schritt einen Merge-Kandidaten aus vorherigen Zweigen — genau der Mechanismus, der abgesetzte Fragmente an den Hauptbaum anbindet statt sie zu verlieren; auch isoliert als Rekonnektions-Heuristik nachimplementierbar. Zweitens liefert ein Online-Konfidenzwert pro Zweig ein fertiges quantitatives QC-Signal. `rivuletpy` ist reines Python, also als zweiter Tracer neben APP2 auf denselben Crops einsetzbar.

### 3.6 Ong, De, Cheng, Ahmed, Yu: *NeuronCyto II: An automatic and quantitative solution for crossover neural cells in high throughput screening.* Cytometry Part A 89(8):747–754, 2016.
DOI: https://doi.org/10.1002/cyto.a.22872 · Software: https://sites.google.com/site/neuroncyto/ (MATLAB) · Priorität: hoch

Das einzige Paper, das exakt deine Situation adressiert — 2D-Fluoreszenz, Zellkultur, High-Throughput, explizit das Crossover-Problem — und die realistische Erwartungshaltung liefert: die Genauigkeit beim Trennen gekreuzter Neuriten steigt von unter 10 % auf **etwa 70 %**, nicht auf 95 %. Zwei übernehmbare Bausteine: (1) Klassifikation jedes Skelettpixels in fünf Typen (Root/Body/Node/Branch/Leaf) mit Tracing ab der Soma-Grenze — liefert unmittelbar deine „Anteil mit Soma-Anschluss"-Metrik; (2) Crossover-Trennung als Label-Propagation auf gerichtetem Graphen (Matrix-Forest-Theorem), robuster als hartes Winkel-Matching, wenn drei oder mehr Neuriten einen Knoten teilen. Liefert außerdem die Begründung, Zellen mit vielen Kreuzungen im QC niedriger zu gewichten.

### 3.7 Loss, Bebis, Parvin: *Iterative Tensor Voting for Perceptual Grouping of Ill-Defined Curvilinear Structures.* IEEE TMI 30(8):1503–1513, 2011.
DOI: https://doi.org/10.1109/TMI.2011.2129526 · Priorität: mittel

Der entscheidende Vorteil gegenüber allen gelernten Reparaturmodellen: **trainingsfrei**, also robust gegen deinen Domain Shift, wo ein auf einer Domäne trainiertes Reparaturnetz kippen kann. Berechne aus dem Rohbild eine Frangi-/Hessian-Vesselness mit kleinen Sigma (1–3 px) plus Hauptrichtung und erzeuge daraus per Stick-Voting eine Saliency-Karte. **Nicht als eigenständige Segmentierung verwenden**, sondern als Zusatzkanal in die Kostenfunktion des Rekonnektionsgraphen: eine Lücke wird nur geschlossen, wenn im Rohbild entlang der Verbindungsstrecke tatsächlich Ridge-Evidenz vorliegt. Das ist dein Schutz gegen falsche Verbindungen zwischen benachbarten, aber unabhängigen Neuriten.

### 3.8 Yang, Li: *Semi-Automatic Correction of 3D Tubular Structure Skeletons via Component-Wise MST and Filtered Delaunay Triangulation.* arXiv:2606.19949, 2026 (to appear, ACM ICMR '26, Amsterdam).
https://arxiv.org/abs/2606.19949 · Priorität: mittel

Übertragbar ist die Kostenfunktion, nicht die Anwendung (3D-Hirngefäße, semi-automatisch, Validierung laut Abstract bisher nur qualitativ, kein Code). Delaunay-Triangulation über die Endpunkte aller Skelettkomponenten verhindert die quadratische Kandidatenexplosion bei tausenden Komponenten; harte Filter nach maximaler Lückenlänge und Winkeltoleranz; Kantengewicht = w₁·Richtungsdiskontinuität + w₂·Distanz + w₃·(negative Ridge-Evidenz); darauf MST oder kürzeste Pfade mit Wurzel im Soma. Der Nutzer-Input-Teil lässt sich bei dir durch die detektierten Somas als automatische Seeds ersetzen. *Jüngste Referenz der Liste — bis zur ACM-DOI als Preprint zitieren.*

### 3.9 Manubens-Gil, Zhou, Chen, … Meijering, Peng et al.: *BigNeuron: a resource to benchmark and predict performance of algorithms for automated tracing of neurons in light microscopy datasets.* Nature Methods 20(6):824–835, 2023.
DOI: https://doi.org/10.1038/s41592-023-01848-5 · Code: https://github.com/Vaa3D/vaa3d_tools · Priorität: mittel
*Author Correction beachten: Nature Methods 2024, DOI 10.1038/s41592-024-02395-3.*

Beantwortet empirisch zwei strategische Fragen. Erstens: über ca. 30.000 Volumina und 35 Algorithmen erklären die **Bildqualitätsmetriken** den größten Teil der Varianz in der Tracing-Qualität, erst danach folgt die Morphologie — deine widersprüchlichen Rollingball-, Belichtungs- und Binning-Effekte sind kein Nebeneffekt, sondern der Haupttreiber. Zweitens: verschiedene Algorithmen liefern komplementäre Information, ein Konsens erhöht die Genauigkeit. Suche also nicht den einen richtigen Tracer, sondern fahre zwei bis drei parallel — die Übereinstimmung zwischen ihnen ist zugleich ein annotationsfreier QC-Score pro Zelle. Die DIADEM-Lehre gilt weiter: automatisches Tracing ersetzt die manuelle Kuration nicht, es priorisiert sie.

---

## 4. Zellauswahl, Zuordnung und Qualitätskontrolle

### 4.1 Li et al.: *Precise segmentation of densely interweaving neuron clusters using G-Cut.* Nat Commun 10:1549, 2019.
→ siehe Top-5, Platz 5.

### 4.2 Kayasandik, Negi, Laezza, Papadakis, Labate: *Automated sorting of neuronal trees in fluorescent images of neuronal networks using NeuroTreeTracer.* Scientific Reports 8:6450, 2018.
DOI: https://doi.org/10.1038/s41598-018-24753-w · Code: https://github.com/cihanbilge/AutomatedTreeStructureExtraction und https://github.com/cihanbilge/SomaExtraction (MATLAB) · Priorität: hoch

Das Setting ist praktisch identisch zu deinem: 2D-Fluoreszenz, Zellkultur, mehrere Zellen, sich berührende und kreuzende Fortsätze, Ziel sind einzelne soma-verwurzelte Bäume. Die Tracing-Routine ist direkt auf dein nnU-Net-Skelett übertragbar und braucht kein neues Netz: dilatierte Soma-Masken definieren Startpunkte **und** Startorientierungen, dann ein gerichtetes Suchfenster, das Punkte in Richtung der lokalen Neuritenorientierung bevorzugt und den Suchwinkel nur ändert, wenn dort kein Seed liegt. Dieser Mechanismus verhindert falsches Abbiegen an Kreuzungen und überbrückt nebenbei kleine Lücken, weil das Suchfenster endliche Reichweite hat. Der Lücken-Radius ist der eine Parameter, den du an deine Fragmentierungsstatistik anpassen musst.

### 4.3 Ong et al.: *NeuronCyto II.* Cytometry A 89(8):747–754, 2016.
→ siehe 3.6. Gehört inhaltlich in beide Abschnitte; die Label-Propagation ist der Zuordnungsteil, die Pixeltyp-Klassifikation der Postprocessing-Teil.

### 4.4 Schmidt, Weigert, Broaddus, Myers: *Cell Detection with Star-convex Polygons.* MICCAI 2018, LNCS 11071, S. 265–273.
DOI: https://doi.org/10.1007/978-3-030-00934-2_30 · arXiv:1806.03535 · Code: https://github.com/stardist/stardist · Priorität: hoch

Löst ein Problem, das deine aktuelle semantische 3-Klassen-Ausgabe strukturell nicht lösen kann: Connected Components auf der Soma-Maske verschmelzen berührende Zellkörper zu einer Instanz — und damit bricht jede nachfolgende Neurit-zu-Soma-Zuordnung, weil die Wurzeln falsch sind. Somas sind rundlich und damit ein Idealfall für StarDist, Neuriten explizit nicht. Empfehlung: zweigleisig fahren statt ein Netz für alles — nnU-Net bleibt für das dünne Skelett, zusätzlich ein StarDist-2D-Modell nur auf dem Soma-Kanal, das eindeutige IDs liefert. Diese IDs sind dann die Wurzeln für G-Cut, NeuroTreeTracer oder die Label-Propagation.

### 4.5 Cutler, Stringer, Lo, Rappez, Stroustrup, Peterson, Wiggins, Mougous: *Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation.* Nature Methods 19(11):1438–1448, 2022.
DOI: https://doi.org/10.1038/s41592-022-01639-4 · Code: https://github.com/kevinjohncutler/omnipose · Priorität: hoch

Der Unterschied zu Cellpose ist genau der, der für dich zählt: Cellpose definiert das Distanzfeld als Distanz zum Zell-**Zentroid**, Omnipose als Distanz zur Zell**grenze**. Die Zentroid-Annahme bricht, sobald ein Fortsatz weiter vom Zellkörper entfernt liegt als der halbe Zelldurchmesser — bei einer Glioblastomzelle mit langen Neuriten ist das der Regelfall. Omnipose liefert Instanz-IDs pro Zelle inklusive Fortsätzen, also die Neurit-zu-Soma-Zuordnung als **Netzausgabe statt als Postprocessing**. Deine 1400 Crops lassen sich zu synthetischen Multi-Zell-Szenen mit bekanntem Overlap zusammensetzen. Zweite Idee unabhängig vom Netz: das grenzbasierte Distanzfeld als zusätzlichen Regressionskopf an nnU-Net hängen.

### 4.6 Zenk, Zimmerer, Isensee, Traub, Norajitra, Jäger, Maier-Hein: *Comparative benchmarking of failure detection methods in medical image segmentation: Unveiling the role of confidence aggregation.* Medical Image Analysis 101:103392, 2025.
DOI: https://doi.org/10.1016/j.media.2024.103392 · arXiv:2406.03323 · Code: https://github.com/MIC-DKFZ/segmentation_failures_benchmark · Priorität: hoch

Deine Frage „welche Zellen sind gut genug", systematisch benchmarkt und explizit **unter Test-Time-Distribution-Shift** — also deinem Fall. Aus derselben Gruppe wie nnU-Net, direkt auf dein 5-Fold-Ensemble anwendbar. Kernbefund: nicht die exotische Unsicherheitsmethode entscheidet, sondern **wie Pixel-Konfidenzen zu einem Objekt-Score aggregiert werden**; der paarweise Dice zwischen Ensemble-Vorhersagen ist die robusteste einfache Baseline. Konkret: pro Zellcrop den paarweisen Score über alle 10 Fold-Paare berechnen, für Klasse 1 mit clDice/Skeleton-Recall statt Dice (sonst dominiert Ein-Pixel-Versatz). Wichtig: **nicht** die mittlere Pixel-Entropie über das ganze Bild nehmen — die riesige Hintergrundfläche deiner Übersichten verwässert sie vollständig; Aggregation auf Vordergrundmaske plus Randregion beschränken.

### 4.7 Griebel, Segebarth, Stein, Schukraft, Tovote, Blum, Flath: *Deep learning-enabled segmentation of ambiguous bioimages with deepflash2.* Nature Communications 14:1679, 2023.
DOI: https://doi.org/10.1038/s41467-023-36960-9 · Code: https://github.com/matjesg/deepflash2 · Priorität: hoch
*Der Preprint arXiv:2111.06693 gehört zur selben Gruppe/Software, trägt aber einen anderen Titel — nicht gleichsetzen.*

Der praktischste QC-Blueprint für echte Fluoreszenzmikroskopie mit mehrdeutigen Bildern: Ensembles plus integrierte Qualitätssicherung mit Unsicherheitsmaßen auf **Instanz-/Regionsebene**, nicht nur pro Bild. Übernimm den Workflow mit deinem eigenen nnU-Net-Ensemble: alle Zellen einer Platte nach Unsicherheit sortieren, nur das obere Perzentil (10–20 %) manuell prüfen, den Rest automatisch akzeptieren — ein kalibrierbares Zeitbudget statt einer Ja/Nein-Schwelle. Zweitens: falls die 1400 Crops von mehreren Personen kuratiert wurden, nutze den Multi-Experten-Teil, um Annotator-Bias in Skelettbreite und -vollständigkeit zu quantifizieren — ein plausibler Mitverursacher der Fragmentierung.

### 4.8 Tomkinson, Bunten, Way: *Stellar quality control for single-cell image-based profiling with coSMicQC.* bioRxiv, 2025.
DOI: https://doi.org/10.1101/2025.10.14.682427 · Code: https://github.com/WayScience/coSMicQC · Priorität: hoch

Modellfreie QC direkt auf den ohnehin extrahierten Morphologie-Merkmalen: z-Score-Normierung ausgewählter Feature-Subsets, Schwellwerte in Standardabweichungen, explizit ausgelegt darauf, **technische Ausreißer von biologisch interessanten Ausreißern zu trennen** — bei Glioblastomzellen mit heterogener Morphologie der kritische Punkt. Erweitere um projektspezifische QC-Features, die deine Fehlermodi direkt abbilden: Komponentenzahl des Skeletts, Größe der größten Komponente relativ zur Gesamtlänge, und vor allem der Anteil somaverbundener Skelettpixel — deine 20–60 % sind bereits eine fertige, interpretierbare QC-Kennzahl. Robuste Schwellen (MAD statt SD) auf einer Referenzcharge, Verteilungen **pro Objektiv/Belichtung getrennt** betrachten, damit Domain Shift nicht als Biologie fehlinterpretiert wird. *(Preprint, kein Peer Review.)*

### 4.9 Lu, Zhao, Xie, Ma, Liu, Zheng: *Quality Control of Neuron Reconstruction Based on Deep Learning.* arXiv:2003.08556, 2020 (Preprint, keine Journalfassung).
https://arxiv.org/abs/2003.08556 · Priorität: hoch

Das einzige gefundene Paper, das QC nicht auf Bild-, sondern auf **Trace-Ebene** macht — also genau auf der Ebene deiner SWC-Traces. QC als binäre Klassifikation **pro Trace-Knoten** umgeht sowohl die riesigen Bildgrößen als auch die komplexe Morphologie und sagt nicht nur, *dass* eine Rekonstruktion schlecht ist, sondern *wo* das Tracing falsch wird. Berichtet: 74,7 % erkannte Fehler bei 1,4 % Fehlalarmen. Drei Nutzen für dich: (1) Anteil fehlerhafter Knoten pro Zelle als QC-Kennzahl, (2) Bruchstellen als Kandidatenliste fürs automatische Gap-Bridging, (3) an Kreuzungen als Plausibilitätsprüfung der Soma-Zuordnung. Labels sind billig, weil die 1400 kuratierten Crops als GT-Traces dienen.

### 4.10 Chen, Murphy: *Evaluation of cell segmentation methods without reference segmentations.* Molecular Biology of the Cell 34(6):ar50, 2023.
DOI: https://doi.org/10.1091/mbc.E22-08-0364 · Code: https://github.com/murphygroup/CellSegmentationEvaluator · Priorität: hoch

Referenzfreie Qualitätsmetriken, per PCA zu einem einzigen Score kombiniert, ausgewertet über 637 Bilder aus vier Modalitäten und 11 Methoden. Eine der stärksten Einzelmetriken ist `FractionOfMatchedCellsAndNuclei`, also die Konsistenz zwischen zwei zusammengehörigen Masken — das strukturelle Analogon zu deiner bereits gemessenen 20–60-%-Größe. **Deine Zahl ist damit kein Nebenbefund, sondern der natürliche Kern eines QC-Scores.** Kandidatenmerkmale: (1) Anteil somaverbundener Skelettpixel, (2) Komponentenzahl, (3) relative Größe der größten Komponente, (4) Intensitäts-/Texturuniformität im Soma, (5) Soma-Flächenplausibilität. Erste PC als Score, Perzentilschwelle auf kuratiertem Satz kalibrieren, Ausschlussquote pro Batch mitberichten.

### 4.11 Valindria, Lavdas, Bai, Kamnitsas, Aboagye, Rockall, Rueckert, Glocker: *Reverse Classification Accuracy: Predicting Segmentation Performance in the Absence of Ground Truth.* IEEE TMI 36(8):1597–1606, 2017.
DOI: https://doi.org/10.1109/TMI.2017.2665165 · arXiv:1702.03407 · Code: https://github.com/vanya2v/Reverse_Classification_Accuracy · Priorität: hoch

Das Referenzpaper für Qualitätsschätzung ohne GT auf **Einzelobjekt-Ebene** — braucht kein Ensemble und keine Unsicherheit, nur ein kleines annotiertes Referenzset, und genau das existiert bei dir bereits. Ablauf: vorhergesagte Maske als Pseudo-Label nehmen, damit einen billigen Reverse-Klassifikator trainieren (Random Forest auf Intensität, Frangi-Filterantworten, Struktur-Tensor-Features — kein zweites nnU-Net nötig), diesen auf 20–50 kuratierten Crops mit echter GT auswerten, den maximal erreichten Score als vorhergesagte Qualität nehmen. Für Klasse 1 Dice durch clDice/Skeleton-Recall ersetzen.

### 4.12 Moshkov, Mathe, Kertesz-Farkas, Hollandi, Horvath: *Test-time augmentation for deep learning-based cell segmentation on microscopy images.* Scientific Reports 10:5068, 2020.
DOI: https://doi.org/10.1038/s41598-020-61808-3 · Priorität: hoch
*Author Correction: Sci Rep 11 (2021), DOI 10.1038/s41598-021-81801-8.*

TTA ist für dich doppelt wertvoll und kostet weder Retraining noch Architekturänderung: die Mittelung über augmentierte Vorhersagen glättet die Wahrscheinlichkeitskarte entlang dünner Strukturen und schließt einen Teil der Lücken, **und** die Streuung liefert gleichzeitig eine Konfidenzkarte für QC. Erweitere die nnU-Net-Inferenz über das eingebaute Mirroring hinaus auf volle 8 Dihedral-Transformationen plus moderate Skalen- und Gamma-Variationen (letztere decken genau deine Belichtungs- und Bittiefenunterschiede ab). Speichere zwei Ausgaben: gemittelte Karte als Segmentierung (vor dem Schwellwert prüfen, ob die Komponentenzahl sinkt) und die pixelweise Standardabweichung als Konfidenzkarte, die anschließend als Kostenbild fürs Gap-Bridging dient.

### 4.13 Wang, Li, Aertsen, Deprest, Ourselin, Vercauteren: *Aleatoric uncertainty estimation with test-time augmentation for medical image segmentation with convolutional neural networks.* Neurocomputing 338:34–45, 2019.
DOI: https://doi.org/10.1016/j.neucom.2019.01.103 · arXiv:1807.07356 · Priorität: mittel
*Bandkorrektur laut Prüfbericht: 338, nicht 335.*

Liefert das *Wie* der Kalibrierung zu Moshkovs *Ob*: TTA wird als Monte-Carlo-Sampling über ein explizites Bildaufnahmemodell formuliert — also über genau die Faktoren, die bei dir zwischen den Chargen variieren. Wähle die Prior-Verteilungen deshalb nicht generisch: Skalierungsbereich aus deinen Objektiven und Binning-Faktoren, Gammabereich aus den beobachteten Belichtungsunterschieden, Rauschmodell aus dem Widefield-vs-Konfokal-Vergleich. Als QC-Metrik pro Zelle dann eine **strukturweite** Größe statt Pixel-Entropie: Variationskoeffizient der Gesamt-Skelettlänge, der Komponentenzahl und der Soma-Fläche über die TTA-Samples.

### 4.14 Zhang, Gao, Lyu, Zhao, Wang, Ding, Wang, Li, Cui: *Characterizing Label Errors: Confident Learning for Noisy-Labeled Image Segmentation.* MICCAI 2020, LNCS, S. 721–730.
DOI: https://doi.org/10.1007/978-3-030-59710-8_70 · Code: https://github.com/502463708/Confident_Learning_for_Noisy-labeled_Medical_Image_Segmentation · Priorität: hoch

1-px-breite Skelettlabels sind der annotationsfehleranfälligste Fall überhaupt — vergessene dünne Neuriten, um 1–2 px versetzte Linien, uneinheitlich behandelte Kreuzungen erzeugen genau das Signal, das ein Recall-lastiger Trainer als „hier ist Hintergrund" lernt. Nutze die Out-of-fold-Wahrscheinlichkeiten aus deiner ohnehin laufenden 5-Fold-CV als Input für cleanlab und ranke die Crops nach Anteil verdächtiger Pixel: die schlechtesten 5–10 % gezielt nachkuratieren schlägt blindes Neuannotieren. **Zwingende Anpassung:** vorher eine Toleranzzone definieren (Label und Vorhersage um 1–2 px dilatieren), sonst meldet CL nur den unvermeidbaren Ein-Pixel-Versatz und ertrinkt in Fehlalarmen.

### 4.15 Talks, Marchesini, Lumetti, Bolelli, Kreshuk: *Unsupervised Source-Free Ranking of Biomedical Segmentation Models Under Distribution Shift.* arXiv:2503.00450, 2025 (Preprint, inzwischen v4).
https://arxiv.org/abs/2503.00450 · Priorität: hoch

Macht dein widersprüchliches Rollingball-Problem endlich **messbar**: Modelle bzw. Preprocessing-Varianten werden allein über die Konsistenz ihrer Vorhersagen unter Perturbationen bewertet, ohne Labels und ohne Zugriff auf Trainingsdaten, und das Ranking korreliert stark mit der echten Target-Domain-Performance. Behandle Rollingball an/aus, uint8- vs. uint16-Normalisierung und Binning-Varianten als konkurrierende Pipelines und ranke sie auf der neuen, unlabelten Charge statt nach Augenmaß. Derselbe Score dient pro Übersichtsbild als Domain-Shift-Warnlampe: fällt er unter das Niveau der Trainingsdomäne, ist Nachannotation fällig, **bevor** überhaupt Morphologie ausgewertet wird.

### 4.16 Gaillochet, Desrosiers, Lombaert: *Active learning for medical image segmentation with stochastic batches.* Medical Image Analysis 90:102958, 2023.
DOI: https://doi.org/10.1016/j.media.2023.102958 · arXiv:2301.07670 · Code: https://github.com/Minimel/StochasticBatchAL · Priorität: mittel

Bei 1400 Crops und begrenzter Kuratierzeit lautet die Frage nicht „mehr Daten", sondern „welche Zellen als nächstes". Naives Top-k-Sampling nach Unsicherheit scheitert bei kleinem Budget, weil die Auswahl von redundanten Beispielen und reinen Ausreißer-Artefakten dominiert wird — die ganze Runde besteht dann aus Debris. Die stochastische Batch-Auswahl ist ein Add-on auf **jede** bestehende Unsicherheitsmetrik, also auf den Ensemble-Score aus 4.6. Stratifiziere den Pool bewusst über die Domänen (Objektiv, Widefield/Konfokal, Bittiefe), damit die Nachannotation den Domain Shift schließt statt die dominante Charge zu verstärken.

### 4.17 Hoffmann, Cho, Zalesky, Di Biase: *From pixels to connections: exploring in vitro neuron reconstruction software for network graph generation.* Communications Biology 7(1):571, 2024.
DOI: https://doi.org/10.1038/s42003-024-06264-9 · Priorität: niedrig

Aktueller Werkzeugvergleich speziell für In-vitro-Neuronenkulturen, mit ausgewiesenen Kernspezifikationen pro Tool. Nutzwert für dich ist eine einzige Unterscheidung: welche Tools überhaupt eine Neurit-zu-Soma-Zuordnung leisten und welche nur aggregierte Feldmaße liefern. NeuriteQuant und HCA-Vision quantifizieren primär auf Bild-/Feldebene und lösen Kreuzungen nicht auf Einzelzellniveau; NeuronCyto II, NeuroTreeTracer und G-Cut zielen explizit auf Einzelzell-Zuordnung. Spart dir, jedes Tool einzeln zu evaluieren.

### 4.18 Pachitariu, Rariden, Stringer: *Cellpose-SAM: superhuman generalization for cellular segmentation.* bioRxiv, 2025.
DOI: https://doi.org/10.1101/2025.04.28.651001 · Code: https://github.com/MouseLand/cellpose · Priorität: mittel
*Stand August 2026 keine peer-reviewte Fassung — vor Abgabe erneut prüfen.*

Die Generalisierung wurde gezielt durch Robustheit gegen Channel-Shuffling, Zellgröße, Schrotrauschen, Downsampling sowie isotropen und anisotropen Blur erhöht — eins zu eins die Liste deiner Störungen (Objektive = Größe und Blur, Belichtung/Bittiefe = Rauschen und Intensitätsskala, Binning = Downsampling, Widefield vs. Konfokal = Blur-Charakteristik). Auch ohne Cellpose-SAM einzusetzen: übernimm die Augmentierungsliste in den nnU-Net-Trainer, denn per Default augmentiert nnU-Net weder mit Schrotrauschen noch mit anisotropem Blur noch mit Downsampling in diesem Umfang. Zusätzlich als Zero-Shot-Baseline für Soma-Instanzen über alle Aufnahmebedingungen laufen lassen, um den Domain-Shift-Anteil am Gesamtfehler überhaupt zu beziffern.

### 4.19 Lalit, Tomancak, Jug: *Embedding-based Instance Segmentation in Microscopy.* MIDL 2021, PMLR 143:399–415. Erweitert als *EmbedSeg: Embedding-based Instance Segmentation for Biomedical Microscopy Data*, Medical Image Analysis 81:102523, 2022.
https://proceedings.mlr.press/v143/lalit21a.html · DOI (Journal): https://doi.org/10.1016/j.media.2022.102523 · Code: https://github.com/juglab/EmbedSeg · Priorität: mittel

Der eleganteste denkbare Ausweg aus der Fragmentierungsfalle: statt Skelettpixel nachträglich über einen Graphen ans Soma zu binden, lernt das Netz pro Pixel ein räumliches Embedding, das auf das Zentrum seiner Instanz zeigt. **Fragmentierung ist damit nicht mehr fatal** — ein isoliertes Fragment zeigt trotzdem auf sein Soma, auch wenn die Pixelkette dorthin unterbrochen ist. Kreuzungen werden ebenfalls entschärft, weil die Zuordnung pro Pixel und nicht über Konnektivität entschieden wird. Umbau: zweiter Ausgabekopf am nnU-Net, der pro Vordergrundpixel den Offset zum Soma-Zentroid regressiert (diskriminativer Loss: Attraktion, Repulsion, Regularisierung), Skeleton-Recall bleibt unverändert auf dem semantischen Kopf. Inferenz per Mean-Shift im Embedding-Raum. Der GPU-Speicherbedarf ist explizit klein gehalten — bei 35000×35000 mit Kachelung relevant.

---

## 5. Evaluation und Morphologie-Merkmalsextraktion

**Merke für diesen ganzen Abschnitt:** Solange Dice die Modellauswahl steuert, optimierst du an deinem Hauptproblem vorbei — Fragmentierung ist in Dice praktisch unsichtbar. Die ersten drei Papers hier sollten gelesen werden, *bevor* du die nächste Trainingsrunde startest, nicht danach.

### 5.1 Maier-Hein, Reinke, Godau et al.: *Metrics reloaded: recommendations for image analysis validation.* Nature Methods 21(2):195–212, 2024.
DOI: https://doi.org/10.1038/s41592-023-02151-z · arXiv:2206.01653 · Priorität: hoch
*Companion: Reinke et al., „Understanding metric-related pitfalls in image analysis validation", Nature Methods 21:182–194, DOI 10.1038/s41592-023-02150-0.*

Liefert das Problem-Fingerprint-Schema zur Metrikauswahl und warnt explizit vor Overlap-Metriken bei sehr dünnen Zielstrukturen — bei 1 px Breite wird Dice fast vollständig von Ein-Pixel-Zentrierungsfehlern dominiert und bestraft Fragmentierung praktisch nicht. Stelle getrennte Fingerprints auf: Klasse 2 (Soma) als semantische Segmentierung mit Overlap plus Boundary-Metrik (NSD), Klasse 1 (Skelett) als curvilineare Struktur mit Topologie-/Zentrallinienmetrik. **Dice für die Skelettklasse künftig nur als Nebenmetrik berichten, nie als Modellauswahl- oder Early-Stopping-Kriterium.** Zusätzlich pro Bild statt gepoolt aggregieren, sonst dominieren die wenigen großen Übersichten die Zahlen.

### 5.2 Shit et al.: clDice — **als Metrik.** CVPR 2021, S. 16560–16569.
→ Zitation siehe 2.4. · Code: https://github.com/jocpae/clDice

clDice ist zuerst eine Metrik und erst danach ein Loss: Skelett der Prädiktion gegen Maske der GT und umgekehrt, harmonisches Mittel. Genau diese Größe bricht ein, wenn dein Skelett in hunderte Komponenten zerfällt, während normaler Dice kaum reagiert — die fehlende Zahl, um dein Hauptproblem messbar zu machen. **Wichtig für dein Setup:** da die GT bereits 1 px breit ist, degeneriert clDice zu reiner Pixel-Precision/Recall — rechne deshalb eine Toleranzvariante (GT-Maske um 1–2 px dilatieren, bevor das Prädiktionsskelett dagegen geschnitten wird), sonst misst du Subpixel-Jitter statt Konnektivität.

### 5.3 Stucki, Paetzold, Shit, Menze, Bauer: *Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes.* ICML 2023, PMLR 202:32698–32727.
https://proceedings.mlr.press/v202/stucki23a.html · arXiv:2211.15272 · Priorität: hoch

Adressiert die Schwäche des reinen Betti-Zahl-Fehlers: der ist global und blind für räumliche Zuordnung, kann also durch zufällige Kompensation (ein falsches Loch hier, ein fehlendes dort) Null werden. Der Betti-Matching-Fehler ist räumlich korrekt und interpretierbar — er sagt nicht nur „hunderte Komponenten statt einer", sondern auch **wo** die falschen Splits sitzen. Als billige Sofort-Surrogate parallel loggen: Komponentenzahl des prädizierten Skeletts, Pixelanzahl der größten Komponente, Anteil somaverbundener Skelettpixel. Diese drei als **Trainingskurve** statt nur Validierungs-Dice machen den Fragmentierungs-Regress überhaupt erst sichtbar.

### 5.4 Mais, Hirsch, Managan, Kandarpa, Rumberger, Reinke, Maier-Hein, Ihrke, Kainmueller: *FISBe: A Real-World Benchmark Dataset for Instance Segmentation of Long-Range Thin Filamentous Structures.* CVPR 2024, S. 22249–22259.
arXiv:2404.00130 · DOI: 10.1109/CVPR52733.2024.02100 · Datensatz: https://doi.org/10.5281/zenodo.10875063 · Projektseite: kainmueller-lab.github.io/fisbe · Priorität: hoch

Das methodisch am nächsten liegende Paper: lange, dünne, weit verzweigte, sich überkreuzende Neuriten mehrerer Zellen in einem Bild, pixelgenau annotiert — mit Metriken, die ausdrücklich für die *nachgelagerte* Analyse aussagekräftig sein sollen: instanzbasierter avF1, Centerline-Recall mit One-to-Many-Matching, average ground truth coverage sowie getrennte Zählung von **False Splits (FS)** und **False Merges (FM)**. FS ist exakt deine Skelettfragmentierung, FM exakt „Neurit an der Kreuzung dem falschen Soma zugeordnet". **Berichte FS und FM getrennt**, denn dein aktueller Trainer (Skeleton 2×) reduziert tendenziell FS und erhöht FM — ein einzelner Score würde diesen Trade-off verstecken. Die average ground truth coverage pro Zelle ist direkt als QC-Schwellwert verwendbar.

### 5.5 Gillette, Brown, Ascoli: *The DIADEM Metric: Comparing Multiple Reconstructions of the Same Neuron.* Neuroinformatics 9(2–3):233–245, 2011.
DOI: https://doi.org/10.1007/s12021-011-9117-y · Priorität: hoch

Vergleicht zwei Rekonstruktionen auf **Trace-Ebene** durch Matching von Bifurkationen und Terminationen samt Topologie — also auf der Ebene, auf der deine SWC-Auswertung stattfindet, nicht auf Pixelüberlapp. Diskontinuitäten und falsche Verzweigungstopologie werden hart bestraft, ein um wenige Pixel versetzter aber topologisch korrekter Ast kaum: exakt die Fehlergewichtung, die du willst. Die XY- und Z-Distanzschwellen beeinflussen das Ergebnis stark — **fixieren und dokumentieren**, sonst sind Zahlen zwischen Runs nicht vergleichbar. Sinnvolle Arbeitsteilung: clDice/Betti auf Pixelebene für das Training, DIADEM auf Trace-Ebene für die Freigabeentscheidung der Pipeline.

### 5.6 Mayerich, Bjornsson, Taylor, Roysam: *NetMets: software for quantifying and visualizing errors in biological network segmentation.* BMC Bioinformatics 13(Suppl 8):S7, 2012.
DOI: https://doi.org/10.1186/1471-2105-13-S8-S7 · Priorität: mittel

Leitet Precision und Recall aus Punkt-zu-nächster-Kurve-Distanzen ab, **getrennt für geometrischen Fehler und Konnektivitätsfehler**. Für dich ist genau diese Trennung der Kern: deine Geometrie ist vermutlich gut (das Netz liegt an der richtigen Stelle), die Konnektivität ist katastrophal — eine einzelne Overlap-Zahl mischt beides und zeigt deshalb nichts. Anders als DIADEM setzt NetMets keinen Wurzelknoten und keine Baumstruktur voraus, funktioniert also auch bei kreuzenden Neuriten mehrerer Zellen mit Zyklen. Die farbcodierte Fehlervisualisierung ist als QC-Overlay für die manuelle Sichtung direkt übernehmbar und weit informativer als eine Differenzmaske.

### 5.7 Arshadi, Günther, Eddison, Harrington, Ferreira: *SNT: a unifying toolbox for quantification of neuronal anatomy.* Nature Methods 18(4):374–377, 2021.
DOI: https://doi.org/10.1038/s41592-021-01105-7 · Code: https://github.com/morphonets/SNT · Priorität: hoch

Schließt die Lücke zwischen nnU-Net-Maske und Feature-Tabelle: Tracing, Proof-Editing, SWC-Import/Export, skriptbar aus Python/Groovy. Nutze es **nicht als Segmentierer**, sondern als Backend für alles nach der Zuordnung: Skelett plus Soma-IDs in SNT-Bäume überführen, am Soma verwurzeln, als SWC exportieren, dann batchweise Sholl-Profile und Strahler-Ordnung statt Eigenbau. Besonders wertvoll ist ein Nebeneffekt: rechne Sholl zusätzlich direkt auf dem Bitmap (Ferreira et al., Nature Methods 11:982–984, 2014, DOI 10.1038/nmeth.3125, inzwischen Teil von SNT) auf demselben Crop — **die Differenz zwischen Bitmap-Sholl und SWC-Sholl ist ein Fragmentierungs-Detektor ohne jede Ground Truth.** Das Proof-Editing-GUI ist zudem der pragmatischste Weg, den Gold-Standard zu erzeugen, gegen den DIADEM und clDice überhaupt erst gerechnet werden.

### 5.8 Scorcioni, Polavaram, Ascoli: *L-Measure: a web-accessible tool for the analysis, comparison and search of digital reconstructions of neuronal morphologies.* Nature Protocols 3(5):866–876, 2008.
DOI: https://doi.org/10.1038/nprot.2008.51 · Priorität: hoch

Definiert das kanonische Vokabular quantitativer Morphometrie aus SWC-Dateien (Gesamtlänge, Verzweigungsordnung, Bifurkationswinkel, Tortuosität, Partition Asymmetry, Terminationszahl). Erfinde kein eigenes Ad-hoc-Feature-Set, sondern docke an Definitionen an, die mit der NeuroMorpho.Org-Literatur vergleichbar sind — bei Glioblastom-Morphologie ist Vergleichbarkeit der einzige Weg, Effekte gegen Literaturwerte zu plausibilisieren. Zweitnutzen: die im Protokoll beschriebene gefilterte Selektion über Boolesche Kombinationen von Morphometriemaßen ist direkt ein automatischer QC-Filter — unplausible Gesamtlänge, unplausible Terminationszahl oder extreme Tortuosität sind fast immer Fragmentierungsartefakte.

### 5.9 Arevalo, Su, Ewald, van Dijk, Carpenter, Singh: *Evaluating batch correction methods for image-based cell profiling.* Nature Communications 15:6516, 2024.
DOI: https://doi.org/10.1038/s41467-024-50613-5 · Preprint: bioRxiv 2023.09.15.558001 · Priorität: hoch

Dein Domain Shift landet nicht nur in der Segmentierung — er zieht sich in die Morphologie-Feature-Tabelle durch und verfälscht jeden Gruppenvergleich. Das Paper benchmarkt zehn Batch-Korrekturverfahren und trennt methodisch sauber vier Metriken für Batch-Entfernung von sechs für Erhalt biologischer Varianz: genau die Doppelbuchführung, die verhindert, dass du mit der Korrektur den Effekt gleich mit wegrechnest. Harmony und Seurat-RPCA lagen in allen fünf Szenarien in den Top 3. Versieh jede Zeile deiner Feature-Tabelle mit expliziten Batch-Labels (Mikroskop, Objektiv, Session, Bittiefe, Preprocessing-Variante) — dann lässt sich der widersprüchliche Rollingball-Abzug als eigener Batch-Faktor **testen** statt heuristisch entschieden zu werden.

---

## Was die Literatur nicht löst

Sieben Punkte, bei denen die gefundenen Arbeiten dir nicht abnehmen, was du selbst entscheiden musst.

**1. Das 1-Pixel-Label als Repräsentationsentscheidung.**
Das ist die größte Lücke. Jede zitierte Topologie-Arbeit — clDice, Topograph, Betti-Matching, Skeleton Recall — setzt Masken **mit Dicke** voraus, aus denen ein Skelett *abgeleitet* wird. Bei dir ist das Skelett bereits das Label. clDice degeneriert dadurch zu Pixel-Precision/Recall, Skeleton Recall verliert seinen Toleranzmechanismus, soft-skeletonization hat nichts mehr zu erodieren. Kein Paper der Liste behandelt den Fall „GT ist bereits das Skelett" als eigenes Problem. Die naheliegende Antwort — die Neuriten stattdessen mit ihrer echten Breite (2–4 px) annotieren oder das 1-px-Label kontrolliert dilatieren und die Zentrallinie erst nachträglich extrahieren — findest du in keiner dieser Arbeiten begründet oder evaluiert. Das musst du selbst als Ablation fahren, und es ist plausibel der größte Einzelhebel überhaupt.

**2. Glioblastom-Morphologie statt Neuronenmorphologie.**
Sämtliche Tracing- und Zuordnungsarbeiten (G-Cut, APP2, Rivulet2, NeuroTreeTracer, DIADEM, L-Measure, BigNeuron) stammen aus der Neuronenrekonstruktion und tragen deren Prioren: baumartige Topologie, ein dominanter Fortsatz, charakteristische Bifurkationswinkel, keine Zyklen. Glioblastomzellen in Kultur sind weder verlässlich baumartig noch in ihrer Winkelstatistik mit kortikalen Neuronen vergleichbar. Das Growth-Orientation-Prior von G-Cut stammt aus über 70.000 NeuroMorpho-Rekonstruktionen — das ist für dich schlicht die falsche Verteilung. Auch die L-Measure-Merkmalsdefinitionen sind für Neuronen normiert; welche davon bei Glioblastom überhaupt biologisch interpretierbar sind, sagt dir kein Paper. Ob deine Zellen SWC-Bäume überhaupt korrekt repräsentieren, ist eine offene Frage, die vor der Merkmalsextraktion beantwortet werden muss.

**3. Der Skalensprung vom 90×89-Crop zum 35000×35000-Übersichtsbild.**
Die nnU-Net-Revisited-Arbeit benennt Patch-Size-Skalierung, aber niemand behandelt den konkreten Fall „Training auf isolierten Einzelzell-Crops, Inferenz auf riesigen Szenen mit vielen sich berührenden Zellen". Der Verteilungsunterschied ist fundamental: in einem kuratierten Crop existiert genau eine Zelle und alles andere ist definitionsgemäß Hintergrund; im Übersichtsbild ist der „Hintergrund" voller echter Neuriten anderer Zellen. Das ist kein Domain Shift im Sinne von BigAug (andere Intensitäten) — es ist ein **Label-Semantik-Shift**, und dafür bietet die Liste nur Park et al. (2.10) mit dem ignore-Label als Teilantwort. Wie du die Crops kontextreicher machst, ohne die Kuratierung zu wiederholen, steht nirgends.

**4. Kachelgrenzen bei Sliding-Window-Inferenz.**
Ein Neurit, der über eine Kachelgrenze läuft, ist ein natürlicher Kandidat für eine Diskontinuität — und keines der Rekonnektionspapers (Carneiro-Esteves, Du, Yang/Li, Tensor Voting) behandelt Kachelartefakte als eigene Fehlerquelle mit eigener Statistik. BaSiC (1.2) adressiert Shading, das ein Teil davon ist, aber nicht den Rest (Gaussian-Weighting am Patch-Rand, Overlap-Wahl, Softmax-Mittelung über Patches). Prüfe die Position deiner Bruchstellen relativ zum Kachelraster — falls sie dort clustern, ist das ein Implementierungsproblem, kein Modellproblem, und die gesamte Loss-Literatur ist dafür irrelevant.

**5. Kalibrierte Rechenkosten.**
Kein Paper sagt dir, was Topograph, Betti-Matching oder ein zweistufiges Rekonnektionsnetz auf 35000×35000-Bildern in deiner Umgebung tatsächlich kosten. Die Laufzeitangaben (Topograph 3–6× schneller als persistente Homologie, MS-LSDNet 0,013 s pro Bild) beziehen sich auf Retina-Bilder in der Größenordnung 1000×1000. Die Extrapolation auf deine Bildgrößen ist offen, und bei einigen Topologie-Losses ist sie nicht linear.

**6. Die Reihenfolge und Interaktion der Maßnahmen.**
Du hast jetzt über vierzig einzeln plausible Vorschläge. Kein Paper sagt, welche sich gegenseitig aufheben. Konkrete Kollisionen: Toleranzband im Loss (2.3) versus Confident Learning auf Labelfehlern (4.14) — beide behandeln denselben Ein-Pixel-Versatz, aber gegensätzlich (einmal tolerieren, einmal als Fehler melden). Oder: TTA-Glättung (4.12) schließt Lücken *und* verwischt Kreuzungen, was FS senkt und FM erhöht, ohne dass eine gemeinsame Metrik das sichtbar machte — außer du hast FISBe (5.4) vorher implementiert. Baue die Metrik zuerst, dann ändere eine Sache zur Zeit.

**7. Wie gut „gut genug" ist.**
Die QC-Papers (4.6–4.11, 5.9) geben dir Scores und Ranglisten, aber keine Antwort auf die eigentliche Frage: ab welchem Fragmentierungsgrad wird deine Morphologie-Auswertung biologisch falsch? NeuronCyto II liefert die einzige belastbare Kalibrierung überhaupt — 70 % Genauigkeit bei Kreuzungen ist der Stand der Technik, nicht 95 %. Was 70 % Zuordnungsgenauigkeit für einen Gruppenvergleich von Neuritenlängen bedeuten, musst du über eine Sensitivitätsanalyse selbst bestimmen: künstlich Fragmentierung in korrekte GT-Traces einbauen und beobachten, ab wann dein Effekt verschwindet oder sich umkehrt. Diese Rechnung findet sich in keinem der Papers, ist aber die Zahl, an der die Freigabe deiner Pipeline hängt.
