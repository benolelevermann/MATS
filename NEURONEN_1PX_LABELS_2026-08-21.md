# 1-px-Skelette als Trainingslabel: was das Neuronen-Feld wirklich tut

## Direkte Antwort

**Ja, solche Verfahren existieren — aber sie sind die Ausnahme, und im Neuronenfeld gibt es genau eine.** Von 26 geprüften Methodenarbeiten trainieren **drei** wirklich binär auf einem 1 Pixel breiten Label: **NeuroFly** (WACV 2025, Neuronen, Ganzhirn), **CP-loss** (IROS 2021, Bordsteine in Luftbildern) und **ThinCrack U-Net** (CACAIE 2023, Risse). Eine vierte (Xu et al. 2021, Risse) behält die 1-px-Annotation, verwandelt sie aber vor dem Training per Gauß-Filter in ein weiches Ziel.

Entscheidend ist, was diese drei zusätzlich tun — denn keine von ihnen trainiert mit Dice/CE allein:

- NeuroFly tauscht **Dice gegen clDice** aus,
- CP-loss gewichtet CE und Dice **ortsabhängig an den Bruchstellen** hoch,
- ThinCrack U-Net verändert **die Architektur** (Down-/Upsampling-Tiefe) und stellt ausdrücklich fest, dass die Loss-Wahl erst *danach* keine Rolle mehr spielt.

Alle übrigen 23 Arbeiten verlassen das 1-px-Regime, bevor trainiert wird: sie dilatieren pauschal auf 2–7 px, regressieren ein weiches Distanz- oder Vektorfeld, oder geben die dichte Segmentierung ganz auf (Punktmengen, Graphen, Tracker). Und in **jeder einzelnen** Neuronenarbeit wird der Zusammenhang außerhalb des Segmentierungsnetzes hergestellt — durch Tracer, Baummodelle, Pfadsucher oder gelernte Verbindungsmodule. Niemand erwartet vom Netz, dass seine Ausgabe eine einzige Komponente ist.

Der Messwert 3,72 Komponenten bei Dice 0,915 auf Trainingsdaten ist damit kein Trainingsdefekt, sondern die erwartbare Eigenschaft der gewählten Repräsentation.

---

## Wie das Feld seine Labels baut

Sortiert: direkt übertragbar (radiusfrei) zuerst. `[LK nicht bestätigt]` markiert Arbeiten, bei denen der Prüfer die im Rechercheentwurf behauptete Label-Konstruktion **nicht** oder **nur teilweise** bestätigen konnte.

### Trainingsverfahren

| Arbeit | Label-Konstruktion | Dilatation | Loss | Übertragbar |
|---|---|---|---|---|
| **NeuroFly** (WACV 2025) | **Echtes 1-Voxel-Skelett**, hellster Pfad zwischen zwei Endpunkten; im Code verifiziert (`mask[coords]=1`, keine Verdickung) | **0 px** | **clDice**, 5 Iter. Soft-Skeletonization, vanilla 3D-U-Net | direkt |
| **CP-loss** (IROS 2021) | **Echtes 1-px-Label** aus Polylinien ohne Breiteninfo. Wörtlich: *„the ground-truth label is already of one-pixel-width, so GT=SkelG and the skeletonization on GT can be omitted"* | **0 px** (weiche Abstandsgewichtung statt Verdickung) | gewichtete CE + gewichteter Dice; Gewichte aus EDT zu detektierten Bruchstellen | direkt |
| **ThinCrack U-Net** (CACAIE 2023) | **Echtes 1-px-Label** für real breitere Risse | **0 px** im Label; Toleranz d=5 px nur in der Metrik | 6 Losses verglichen — *nach* dem Architekturumbau kein signifikanter Unterschied | direkt |
| **Xu et al., Thin Crack** (arXiv 2101.03326) | 1-px-Annotation, danach **Gauß-geglättet** | 0 px im Label; 3-px-Scheibe nur für TP/FP in der Metrik | **MSE statt BCE** | direkt |
| **Mnih & Hinton** (ECCV 2010) | 1-px-Zentrallinie → Gauß-Heatmap `M=exp(-d²/σ²)`. Datenlage identisch: *„no information about road widths"* | σ = 2 px (≈5 px weicher Saum) | CE gegen reellwertiges Ziel; Metrik relaxed, ρ=3 px | direkt |
| **Li et al.** (Front. Neuroinf. 2025) | SWC → Resampling ≤1 Voxel → 3-Voxel-Kugelträger → **Distanzfeld** | 3 Voxel pauschal, **Radiusspalte ungenutzt** | **L1**, regionengewichtet | direkt |
| **PointTree / Cai et al.** (eLife 2025) | dito; im Code verifiziert, dass nur Spalten 2:5 (Koordinaten) gelesen werden | 5 Voxel (Paper) / 7×7×7-Kernel (Code) | **L1** auf Distanzfeld, UNet3D | direkt |
| **Oner et al.** (TMI 2022) | Zentrallinien ohne Radius → **trunkiertes Distanzfeld**; Annotation wird als Snake mitkorrigiert | keine; Trunkierung `dmax` — `[LK nicht bestätigt]`: die „2 Voxel" des Entwurfs sind falsch, der Code setzt **dmax = 15** | MSE + Snake-Regularisierer (**kein** separater Konnektivitätsterm — Entwurf korrigiert) | direkt |
| **CAPE** (MICCAI 2025) | Graph aus der Annotation; dichte Basis = Distanzfeld | keine (10 px nur als Suchkorridor **im Loss**) | MSE + Dijkstra-**Pfadkosten** zwischen GT-Knotenpaaren | direkt |
| **Skeleton Recall Loss** (ECCV 2024) | Skelett der GT-Maske, 2× dilatiert, dann `skel *= seg_all` **zurück auf die GT maskiert** | 2 px Diamant — bei 1-px-GT durch die Rückmaskierung **effektiv 0** | Dice + CE + Soft-**Skeleton-Recall** (Nenner nur GT) | direkt (nnU-Net-v2-Fork) |
| **DeepFlux** (CVPR 2019) | 1-px-Skelett → 7-px-Kontextband; Ziel = Einheits**vektor** zum nächsten Skelettpixel | 7 px, nur Träger des Vektorfelds, kein Klassenlabel | klassenbalancierter L2 | direkt |
| **Huang et al.** (Front. Neuroanat. 2020) | Zentrallinie → 3D-Zylinder, **fester** Radius 2 px | 2 px pauschal | gewichtete CE + Dice (σ=0,5) | direkt |
| **DeepNeuron** (Brain Inf. 2018) | **Gar kein Pixel-Label**: Patch- und Patchpaar-Klassenlabel aus Knoten + Parent-Relation | n/a | CE (Detektion) + **Contrastive Loss** (Siamese-Verbindung; Entwurf sagte CE) | direkt |
| **Topograph** (ICLR 2025) | Standardmasken; Loss **breitenagnostisch** über Komponentengraph (TP/TN/FN/FP) | intern ε / 2ε **im Loss** | Dice/CE + Komponentengraph-Term | mit Anpassung |
| **clDice** (CVPR 2021) | **Setzt dicke Masken voraus** und skelettiert zur Laufzeit; k ≥ maximaler Pixelradius | keine (erodiert statt dilatiert) | (1−α)(1−softDice) + α(1−softclDice) | mit Anpassung |
| **Sat2Graph** (ECCV 2020) | Graph-Tensor: Vertexness (3×3 je Knoten) + 6 Richtungsvektoren; die Kante wird **nie als Linie gemalt** | r=1 nur als Lokalisierung | Softmax-CE + L2 auf Versatzvektoren | mit Anpassung |
| **Relationformer** (ECCV 2022) | Reines Graphenlabel; Richtung ist Graph → Maske, nicht umgekehrt | keine | DETR-Set-Loss + Relationsklassifikation | mit Anpassung |
| **PointNeuron** (WACV 2023) | Punktwolke aus Schwellwert; SWC-Knoten als Ziel; Adjazenz per Graph-Autoencoder | keine | Chamfer + Objectness-CE + Radius-MAE (streichbar) + maskierte CE auf Adjazenz | mit Anpassung; **Code trotz Repo nicht vorhanden** |
| **Wolterink et al.** (MedIA 2019) | Kein Rasterlabel; Patch → **Richtungsklasse** auf der Kugel (|D|=500) | n/a | CE über Richtungen + Radiusregression (streichbar) | mit Anpassung |
| **Sironi et al.** (CVPR 2014) | Distanzfeld-Regression | Toleranzband d_M = s/2 | quadratischer Verlust, GradientBoost | `[LK nicht bestätigt]` — die evaluierte **Multiskalen**-Variante nimmt Radien als Bestandteil der Annotation; nur die Fixradius-Variante ist radiusfrei → **mit Anpassung**, nicht „direkt" |
| **SoftSeg** (MedIA 2021) | Weiche Labels aus Interpolation; kein Zentrallinienverfahren | n/a | Adaptive Wing Loss + normalisiertes ReLU | mit Anpassung |
| **DeepBranchTracer** (AAAI 2024) | Zwei Label aus derselben Punktmenge, Centerline mit r=1 | `[LK nicht bestätigt]` — **r=1 ist ein RADIUS, kein Durchmesser**: das „Centerline-Label" ist eine 2–3 px breite Röhre, ~5–9× mehr Vordergrund als ein echtes 1-px-Skelett | gewichtete BCE + MSE (Radius) + CE/Kosinus (Richtung) | mit Anpassung (Radiuskopf entfällt) |
| **NeuroLink** (MICCAI 2024) | `[LK nicht bestätigt]` — Dilatationsradius **nirgends beziffert**; das behauptete „Ignore-Label" **existiert nicht**: ignoriert werden nur die ersten zwei Ringe im FN-Term der WLC-Loss | unbekannt | **Tversky (γ=0,7)** + WLC + Richtungs-MSE + Distanz-MSE (Entwurf ließ Tversky ganz aus) | mit Anpassung |
| **3D WaveUNet** (Bioinformatics 2022) | `[LK widerlegt]` — Code liest die SWC-**Radiusspalte** (`radius=float(e[5])`) und malt Kugeln mit r+0,25; das Paper schließt radiuslose Rekonstruktionen **ausdrücklich aus dem Datensatz aus** | radiusabhängig | gewichtete CE (1 / 5) | **setzt Radius voraus** |
| **NRTR** (TMI 2024) | SWC als Punktmenge, keine Rasterisierung | n/a | Set-Loss: CE + L1 + **GIoU auf (Zentrum, Radius)-Box** | **setzt Radius voraus** |
| **He et al.** (MICCAI 2020) | Zentrallinien-Heatmap mit Abfall **am Gefäßradius**; zwei **getrennte** Netze (nicht zwei Köpfe — Entwurf korrigiert) | radiusabhängig | BCE+Dice / MSE + Minimal-Cost-Path | **setzt Radius voraus** |

### Werkzeuge, Benchmarks, Metriken (kein Training)

| Werkzeug | Label-Konstruktion | Dilatation | Relevanz |
|---|---|---|---|
| **SNT** `Tree.skeletonize()` | 1-px-Bresenham, Radius wird nie gelesen. Javadoc: *„1-pixel-wide Bresenham rasterization **suitable for topological analysis**"* | 0 px | **Der Weg, den der Nutzer gegangen ist** — SNT deklariert ihn für die Auswertung, nicht fürs Training |
| **SNT** `util.TreeToRaster` | Kegelstümpfe mit interpolierten Radien, 5×-Supersampling, Brücken-Frustum an Gabelungen | Default `lateralRes/2` wenn kein Radius; `setRadiusScale()` skaliert frei | Der von SNT selbst vorgesehene andere Pfad; existiert erst ab ca. SNT 5.0.12 (Juni 2026) |
| **Vaa3D** `swc_to_maskimage` | Kugel je Knoten vom Knotenradius | keine | Bricht bei r≤0 ab: *„You have illeagal radius values."* — das Feld stuft Radius 0 als **Datenfehler** ein |
| **neuTube / NeuTu** | Geo3d_Ball vom Knotenradius + Tubussegment zum Parent | keine | dritter unabhängiger Beleg für radiusbasierte Rasterisierung |
| **navis** `voxelize()` | binnt **nur Knotenkoordinaten**, keine Kanteninterpolation | keine | Warnung: erzeugt aus einer SWC ohne Resampling eine **zerfallene Punktwolke** |
| **BigNeuron** + `neuron_radius` | Keine Rasterisierung; Bewertung nur im SWC-Graphraum. Fehlt der Radius: *„we estimated the radius of the trees using the 'neuron_radius' plugin"* | n/a | **Der kanonische Umgang mit fehlendem Radius: aus dem Bild schätzen.** Eigene 2D-Variante `markerRadiusXY()` bei `sz[2]==1` |
| **DIADEM / PyNeval** | Keine — SWC-Graph gegen SWC-Graph | n/a | Im Feld gibt es weder Pixel-Dice noch Komponentenzählung als Zielgröße |

---

## Was bei fehlendem Radius funktioniert

Die Radiusspalte ist ein **deutlich kleineres Hindernis als erwartet**. Es fallen genau vier Verfahren und drei Werkzeuge weg:

**Raus:** 3D WaveUNet (Code liest die Spalte; das Paper wirft radiuslose SWC aus dem Datensatz), NRTR (GIoU-Term auf der Radius-Box degeneriert bei r=0 zum Punkt), He et al. MICCAI 2020 (Heatmap-Abfall am Gefäßradius), Sironis Multiskalen-Variante, sowie Vaa3D `swc_to_maskimage` (bricht ab), neuTube und `swc2mask_cylinder` (degeneriert bei r=0 wieder zur 1-Voxel-Linie). Bei DeepBranchTracer und Wolterink entfällt nur je ein additiver Kopf/Term, nicht das Verfahren.

**Bleibt:** alles andere — 19 von 26 Arbeiten. Und das ist kein Zufall. Das Feld **ignoriert die Radiusspalte auch dort, wo sie vorhanden wäre**:

- Huang et al. 2020: pauschal 2 px für alle Neuriten, obwohl die SWC Radien hat
- Li et al. 2025: pauschal 3 Voxel, Radiusspalte im ganzen Paper nicht erwähnt
- PointTree: im Code verifiziert, dass die Radiusspalte übersprungen wird
- Skeleton Recall Loss: pauschal 2 px Diamant-Kernel

Der Satz „für Deep Learning braucht man die Radiusinformation aus der SWC" ist empirisch falsch. **Radius 0.000 ist kein Ausschlusskriterium.**

Es gibt zusätzlich zwei Wege, den fehlenden Radius zu *beschaffen*, statt ihn zu ersetzen:

1. **Aus dem Bild schätzen** — der kanonische BigNeuron-Weg. Das Vaa3D-Plugin `neuron_radius` lässt um jeden Knoten konzentrische Schalen wachsen und stoppt, wenn der Anteil unterschwelliger Pixel eine Toleranz überschreitet. Es hat mit `markerRadiusXY()` eine eigene 2D-Variante, die bei `sz[2]==1` automatisch greift. Die Glioblastom-Fortsätze *haben* im Bild eine messbare Breite — nur die SNT-SWC hat sie nicht gespeichert.
2. **In SNT selbst neu rendern** — dieselbe SWC mit `TreeToRaster` statt `Tree.skeletonize()`, mit gesetztem `setDefaultRadius()` oder `setRadiusScale() > 1`. Das ist ein Parameter, keine Neuimplementierung. Achtung: `Path.hasRadius()` ist als `radius > 0d` definiert — eine durchgehende 0.000 löst den Fallback (halbe Voxelbreite) **stillschweigend und ohne Fehlermeldung** aus.

---

## Die übertragbaren Verfahren im Einzelnen

### 1. NeuroFly — das einzige Neuronen-Verfahren, das wirklich auf 1 Voxel trainiert

**Zitation:** Zhao R., Liu Y., Zhang S., Yi Z., Xiao Y., Xu F., Yang Y., Zhou P. (2025). *NeuroFly: A framework for whole-brain single neuron reconstruction.* WACV 2025.
**Preprint:** arXiv:2411.04715 · **Code:** github.com/beanli161514/neurofly

**Mechanismus.** Das Label ist ein 1-Voxel-Pfad, im Code doppelt belegt (`skel_annotator.py`: `mask = np.zeros(...)`, dann `mask[coords]=1`; `aug_segs.py` kombiniert nur per `np.clip(mask1+mask2,0,1)`). Kein Radius, keine Dilatation. Der gesamte Hebel liegt im **Verlust**: clDice mit 5 Iterationen Soft-Skeletonization auf einem vanilla 3D-U-Net (3 Ebenen, 32/64/128). Die Begründung zitieren die Autoren wörtlich aus Shit et al.: die differenzierbare Skelettierung erlaube *„end-to-end training … using only skeleton labels instead of mask labels"*.

Warum das gegen die 3,72 Komponenten wirkt: Dice sieht einen fehlenden Pixel auf einem 1-px-Label als Fehler im Promillebereich — und genau dieser Pixel zerreißt die Komponente. clDice bestraft ihn hart, weil ein unterbrochenes Vorhersageskelett das Referenzskelett nicht mehr überdeckt.

**Ehrliche Einschränkung.** Die theoretische clDice-Analyse (siehe unten) sagt, dass der Präzisionsterm `Tprec = |S_P ∩ V_L|/|S_P|` auf einem 1-px-GT subpixelgenaue Lage verlangt und deshalb unbrauchbar streng sein müsste. NeuroFly funktioniert trotzdem produktiv im Ganzhirnmaßstab. Die Auflösung ist vermutlich, dass die Fasern im Bild ~3 Voxel breit sind und die Vorhersage dadurch ohnehin auf der Achse zentriert liegt. Für ein 2D-Fluoreszenzbild ist das nicht garantiert — **erst α klein wählen (0,1–0,2) und die Kurve beobachten**, nicht blind α=0,5 setzen.

**Konkret in diesem Projekt.** clDice als Zusatzterm zum bestehenden Dice+CE in einen nnU-Net-Custom-Loss einbauen, nur auf Klasse 1. Referenzimplementierung github.com/jocpae/clDice. **Codefalle:** in `cldice_loss/pytorch/cldice.py` wird das Argument `iter_` zwar gespeichert, der Skelettierer aber hart als `SoftSkeletonize(num_iter=10)` instanziiert — das Argument hat keine Wirkung.

---

### 2. Skeleton Recall Loss — bereits ein nnU-Net-v2-Trainer

**Zitation:** Kirchhoff Y., Rokuss M.R., Roy S., Kovács B., Ulrich C., Wald T., Zenk M., Vollmuth P., Kleesiek J., Isensee F., Maier-Hein K.H. (2024). *Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures.* ECCV 2024.
**DOI:** 10.1007/978-3-031-72980-5_13 · arXiv:2404.03010 · **Code:** github.com/MIC-DKFZ/Skeleton-Recall
*(Die beiden Prüfberichte widersprechen sich beim LNCS-Band — 15135 vs. 15142. Die DOI ist eindeutig.)*

**Mechanismus.** `L = Dice + CE + w·SoftSkeletonRecall`, mit `rec = Σ(p·skel) / Σ(skel)`. Der Nenner enthält **nur das GT**, nie die Vorhersage: der Term belohnt das Treffen der Zentrallinie und bestraft Überbreite **nicht**. Genau die Asymmetrie, die ein 1-px-Label braucht — Überdeckung ist erzwingbar, subpixelgenaue Lage nicht. Gleiche Gruppe wie nnU-Net, also ein Trainer-Austausch statt einer Neuimplementierung, und mehrklassenfähig.

**Zwei Fallen, beide im Code verifiziert.**

1. `SkeletonTransform` macht `skel = skeletonize(bin_seg)`, dann `dilation(dilation(skel))`, dann `skel *= seg_all[0]`. Die letzte Zeile maskiert den Schlauch **zurück auf die GT-Maske**. Bei einem 1-px-Label ist `skeletonize` die Identität und die Rückmaskierung frisst den Toleranzschlauch komplett auf: `skel == GT`. Der Verlust läuft, verschenkt aber genau das Band, wegen dem er wirkt. Die Zeile ersatzlos zu löschen reicht **nicht** — sie stellt die Klassenindizes wieder her; ohne sie ist `skel` binär 0/1 und der `scatter_` ordnet den Schlauch pauschal Klasse 1 zu. Korrekt ist: dilatieren, dann den Schlauch per Nearest-Neighbour aus der GT neu klassenbeschriften.
2. `skeletonize` läuft auf `bin_seg = (seg_all > 0)`, also auf der **Vereinigung von Skelett und Soma**. Der Soma-Blob wird dadurch zu einer Linie ausgedünnt, und an der Soma-Neurit-Verbindung kann sich die Medialachse gegenüber dem annotierten Skelett verschieben. Im 3-Klassen-Setup ist das kein Randfall, sondern der Normalfall.

**Ehrliche Einschränkung.** Eine Reproduktionsstudie (Arora, Kumar, Gupta, arXiv:2508.11374, 2025) findet, dass SRL-Modelle die Baselines auf mehreren Originaldatensätzen **nicht** übertreffen. Messen, nicht glauben.

---

### 3. CP-loss — der explizite 1-px-Konnektivitätsverlust

**Zitation:** Xu Z., Sun Y., Wang L., Liu M. (2021). *CP-loss: Connectivity-preserving Loss for Road Curb Detection in Autonomous Driving with Aerial Images.* IEEE/RSJ IROS 2021, S. 1117–1123.
**arXiv:2107.11920** · Projektseite: tonyxuqaq.github.io/projects/CP-Loss/ · kein Repo verifiziert, Formel im Paper vollständig

**Mechanismus.** Das ist der einzige Satz im gesamten Feld, der die Frage frontal beantwortet: *„Note that the ground-truth label is already of one-pixel-width, so GT = SkelG and the skeletonization on GT can be omitted."* Skelettiert wird ausschließlich die **Vorhersage**. Pro Trainingsschritt werden die Bruchstellen bestimmt — GT-Skelettpixel ohne Vorhersage (`Skel_fG`) und Vorhersageskelettpixel ohne GT-Entsprechung (`Skel_fP`) — und der CE/Dice-Beitrag wird in deren Umgebung über eine Distanztransformation hochgewichtet: `u_i = [1 + exp(-d(x_i, Skel_fG)/σ) - p_i]²`, mit einem Focal-artigen Quadrat. Kein Radius, keine Dilatation, kein Distanzfeld. Die Herkunft der Labels ist strukturell identisch zum SWC-Fall: Polylinien ohne jede Breiteninformation, zu 1-px-Linien rasterisiert.

**Konkret.** ~50 Zeilen in einem nnU-Net-Custom-Loss, plus pro Batch eine Skelettierung + EDT der Vorhersage auf der CPU. Nur auf Klasse 1 anwenden, Klasse 2 (Soma) beim Standard-Dice/CE lassen. Das ist der Verlust, der am direktesten optimiert, was der Nutzer misst.

---

### 4. Distanzfeld-Regression aus radiusloser SWC — die Quan-Gruppen-Blaupause

**Zitation A:** Li L., Hu Y., Wang X., Sun P., Quan T. (2025). *A neuronal imaging dataset for deep learning in the reconstruction of single-neuron axons.* Frontiers in Neuroinformatics 19:1628030. **DOI:** 10.3389/fninf.2025.1628030 · Daten: zenodo.org/records/15372438
**Zitation B:** Cai L., Fan T., Qu X., et al., Quan T., Zeng S. (2025). *Automatic and accurate reconstruction of long-range axonal projections of single-neuron in mouse brain.* eLife 13:RP102840. **DOI:** 10.7554/eLife.102840.3 · **Code:** github.com/FateUBW0227/Seg_Net (`PreMakeData.py`)

**Mechanismus.** Dies ist die nächstliegende Arbeit zum Fall des Nutzers: SWC-Zentrallinien aus (halb-)manuellem Tracing, kein verwertbarer Radius, Fluoreszenzmikroskopie. Drei Schritte:

1. **Resampling** der Skelettpunkte auf ≤1 Voxel Abstand — verhindert, dass das Label schon beim Rasterisieren zerfällt.
2. **Pauschaler Kugelträger** von 3 Voxeln (Li) bzw. 5 Voxeln / 7×7×7-Kernel im Code (PointTree) um jeden Punkt. Das ist **keine** anatomische Aussage, nur der Träger des Feldes.
3. **Distanzfeld** innerhalb des Trägers, maximal auf der Zentrallinie, abfallend nach außen — trainiert mit **L1**, nicht Dice/CE.

Im PointTree-Code ist zeilenweise belegt, dass nur `item[2:5]` (Koordinaten) und `item[-1]` (Parent) gelesen werden; die Radiusspalte wird übersprungen. Das weiche Feld entsteht per `-np.log(minD + dfImg/maxD*(1-minD))` und wird als uint8-TIFF gespeichert. *(Das Paper nennt es „Gaussian kernel distances", der Code implementiert eine negativ-logarithmische Abbildung — wer es nachbaut, sollte dem Code folgen.)*

**Warum das genau die gemessene Diskrepanz auflöst.** Bei einem binären 1-px-Ziel ist eine um 2 px versetzte Vorhersage ein Totalausfall — falsch positiv *und* falsch negativ. Beim Distanzfeld ist sie ein kleiner L1-Fehler. Das Klassenungleichgewicht verschwindet vollständig, weil kein binäres Ziel mehr existiert, und der Grat eines stetigen Feldes ist von Natur aus zusammenhängend: es fällt nirgends abrupt auf null.

**Konkret.** Klasse 1 in einen Regressionskanal umbauen (Resampling ≤1 px → 3-px-Träger → Distanzfeld → L1), Klasse 2 (Soma) als normale Klassifikation belassen. In nnU-Net erfordert das einen Custom-Trainer mit Regressionskopf ohne Softmax — die invasivste, aber sauberste Variante. Die 1-px-Ausgabe entsteht am **Ende**, durch Kammextraktion oder Skelettierung des geschwellten Feldes, nicht als Trainingsziel.

---

### 5. Mnih & Hinton — dasselbe Problem, 2010 gelöst, in zehn Zeilen

**Zitation:** Mnih V., Hinton G.E. (2010). *Learning to Detect Roads in High-Resolution Aerial Images.* ECCV 2010, LNCS 6316, S. 210–223.
**Volltext:** cs.toronto.edu/~hinton/absps/road_detection.pdf · kein offizieller Code

**Mechanismus.** Die Ausgangslage ist strukturell identisch: *„most road maps come in a vector format that only specifies the centreline of each road and provides no information about road widths."* Die Antwort: 1-px rasterisieren, dann `M(i,j) = exp(-d(i,j)²/σ²)` mit `d` = euklidische Distanz zum nächsten Zentrallinienpixel, σ = 2 px. Trainiert wird CE gegen das **reellwertige** Ziel. Evaluiert wird mit relaxed completeness/correctness bei ρ = 3 px. Die Diagnose steht wörtlich im Paper: *„errors of this type hurt the training because the network is trying to fit inconsistent labels."*

**Konkret.** Das ist der billigste Test der ganzen Liste: `scipy.ndimage.distance_transform_edt` auf dem invertierten 1-px-Label, `exp(-d²/4)`, fertig. Als Soft-Label für Klasse 1 in nnU-Net einspeisen (region-based training oder Custom-Trainer mit Soft-Target-CE). Wenn das die Komponentenzahl schon halbiert, ist die Repräsentation die Ursache und man muss die Loss gar nicht anfassen.

**Wichtige Abgrenzung:** Der Massachusetts-Roads-Datensatz, auf dem clDice & Co. trainieren, stammt aus Mnihs Dissertation 2013 und hat **harte 7-px-Balken**, nicht dieses weiche Ziel. Zwei verschiedene Rezepte desselben Autors.

---

### 6. ThinCrack U-Net — der Architektur-Hebel, den niemand sonst nennt

**Zitation:** Siriborvornratanakul T. (2023). *Pixel-level thin crack detection on road surface using convolutional neural network for severely imbalanced data.* Computer-Aided Civil and Infrastructure Engineering 38(16):2300–2316.
**DOI:** 10.1111/mice.13010 · kein offizielles Repo; CrackTree260 / CRKWH100 / CrackLS315 öffentlich

**Mechanismus.** Die einzige Arbeit, die dicht und binär auf echten 1-px-Labels trainiert und den Fehler **in der Architektur** verortet: *„the repetitive use of downsampling and upsampling in Modified U-Net may cause the thickness."* Bei 5–6 Auflösungsstufen entspricht ein Bottleneck-Pixel 32–64 Bildpixeln — eine 1-px-Zentrallinie existiert dort als Struktur nicht mehr und wird beim Hochsamplen als verwaschener, an dünnen Stellen abreißender Fleck rekonstruiert. Die Lösung: Down-/Upsampling drastisch reduzieren, das verlorene rezeptive Feld über **atrous convolution** zurückholen. F-Measure auf CrackTree260: 65,71 % → 94,48 %.

**Zwei Präzisierungen des Prüfers.** (a) Das Ergebnis „Loss egal" gilt **nur nach** dem Architekturfix — im unkorrigierten U-Net machen die sechs verglichenen Losses sehr wohl einen Unterschied. (b) Vorgeschlagen wird eine *„balanced usage"* weniger Stufen plus atrous conv, nicht das Entfernen aller Stufen (Speicherbedarf).

**Konkret.** nnU-Net wählt die Zahl der Auflösungsstufen automatisch aus der Patchgröße — bei typischen 2D-Patches sind das 6–7. Das steht in `nnUNetPlans.json` unter `n_conv_per_stage` / `pool_op_kernel_sizes` und lässt sich ohne Codeänderung auf 4 Stufen kürzen. Das ist ein reiner Konfigurationstest, kostet einen Trainingslauf und ist unabhängig von jeder Label- oder Loss-Änderung. Angesichts dessen, dass das Problem **auf den Trainingsdaten selbst** auftritt, ist das ein sehr plausibler Kandidat.

---

### 7. Oner et al. — Distanzfeld plus Korrektur der Annotation selbst

**Zitation:** Oner D., Koziński M., Citraro L., Fua P. (2022). *Adjusting the Ground Truth Annotations for Connectivity-Based Learning to Delineate.* IEEE TMI 41(12):3675–3685.
**arXiv:2112.02781** · **Code:** github.com/doruk-oner/AdjustingAnnotationswithSnakes

**Mechanismus.** Gleiche Datenlage (manuell getracte Zentrallinien ohne Radius), gleiche Modalitäten (Zwei-Photonen-Axone, Neuronenstapel). Zwei Bausteine: (a) trunkiertes Distanzfeld als Regressionsziel mit MSE; (b) die Annotation wird während des Trainings als **aktive Kontur** behandelt und topologieerhaltend verschoben, weil handgetracte Zentrallinien systematisch um 1–2 Pixel gegenüber der wahren Struktur versetzt sind.

Baustein (b) ist die wichtigste Diagnose der ganzen Recherche für diesen Fall: bei einem 1-px-Label ist so ein Versatz nicht wegzumitteln. Das Netz lernt widersprüchliche Ziele an leicht verschobenen Stellen und antwortet mit ausgedünnten, unsicheren Vorhersagen — deren Wahrscheinlichkeiten unter 0,5 fallen, und genau dort reißt die Linie. Bei SNT/Fiji-Tracings über Fluoreszenzbilder mit weichen Kanten ist derselbe Versatz zu erwarten.

**Zwei Korrekturen des Prüfers.** Der Trunkierungsradius ist **nicht 2 Voxel** (das ist der Testzeit-Schwellwert auf der Vorhersage) — der Code setzt `dmax: 15`. Und der Verlust enthält **keinen** separaten Konnektivitätsterm aus der Vorarbeit; es ist reines MSE plus Snake-Regularisierer.

**Konkret.** Baustein (a) allein ist identisch mit Abschnitt 4 und sofort umsetzbar. Baustein (b) ist der Zusatzschritt, wenn (a) allein nicht reicht.

---

### 8. CAPE — Pfadkosten statt Pixelmengen, Graph direkt aus der SWC

**Zitation:** Esmaeilzadeh E., Garaaghaji E., Hallaji Azad F., Oner D. (2025). *CAPE: Connectivity-Aware Path Enforcement Loss for Curvilinear Structure Delineation.* MICCAI 2025 (angenommen und veröffentlicht).
**arXiv:2504.00753** · papers.miccai.org/miccai-2025/paper/4309_paper.pdf · **Code:** github.com/NeuraVisionLab/CAPE

**Mechanismus.** `L_total = L_MSE(y, ŷ) + α · L_CAPE(G, ŷ)`. Für Knotenpaare aus dem GT-Graphen wird per Dijkstra der günstigste Pfad **in der Vorhersage** gesucht und dessen Kosten `Σ ŷ(n)²` bestraft. Eine Lücke macht den günstigsten Pfad teuer, unabhängig von der Strukturbreite. Es werden nie Pixelmengen verglichen, damit ist der Verlust strukturell unempfindlich gegen Labelbreite **und** Labelversatz — die beiden Größen, die hier undefiniert bzw. verrauscht sind.

Der besondere Reiz: **die SWC liefert den Graphen frei Haus** (Knoten = Samples, Kanten = Parent-Relation). Nichts muss skelettiert oder geschätzt werden. Und die Zielgröße des Verlusts (Erreichbarkeit zwischen Knoten) ist praktisch identisch mit der Diagnosemetrik des Nutzers.

**Voraussetzung, die man kennen muss:** CAPE steht nicht allein, sondern setzt die MSE-Distanzfeldbasis aus Abschnitt 4/7 voraus. Erst das Distanzfeld, dann CAPE obendrauf. Kosten: ein Dijkstra-Lauf pro Batch.

---

### 9. Radius nachträglich beschaffen — SNT TreeToRaster und BigNeuron `neuron_radius`

**Zitationen:** Arshadi C., Günther U., Eddison M., Harrington K.I.S., Ferreira T.A. (2021). *SNT: a unifying toolbox for quantification of neuronal anatomy.* Nature Methods 18:374–377. **DOI:** 10.1038/s41592-021-01105-7 · github.com/morphonets/SNT
Manubens-Gil L. et al. (2023). *BigNeuron: a resource to benchmark and predict performance of algorithms for automated tracing of neurons.* Nature Methods 20:824–835. **DOI:** 10.1038/s41592-023-01848-5 · Plugin: `vaa3d_tools/.../neuron_radius/marker_radius.h`

**Mechanismus.** SNT trennt selbst zwischen Topologie-Darstellung (`Tree.skeletonize()`, 1 px, radiusfrei, laut Javadoc *„suitable for topological analysis"*) und volumetrischem Rendering (`TreeToRaster`, Kegelstümpfe mit interpolierten Radien, Brücken-Frustum an Gabelungen, 5×-Supersampling). Der Nutzer hat den Auswertungspfad als Trainingspfad benutzt. Mit `setDefaultRadius()` oder `setRadiusScale() > 1` liefert dieselbe SWC ohne jede Radiusinformation eine definiert dicke, an Verzweigungen zusammenhängende Maske — zwei Parameter, keine Neuimplementierung. *(Erfordert SNT ab ca. 5.0.12, Juni 2026.)*

Alternativ der BigNeuron-Weg: den Radius **aus dem Grauwertbild schätzen**. `markerRadius` lässt konzentrische Schalen ab r=1 wachsen und stoppt, wenn der Anteil unterschwelliger Pixel eine Toleranz überschreitet; `markerRadiusXY()` ist die 2D-Variante und greift bei `sz[2]==1` automatisch. *(Die Default-Methode ist `markerRadius_hanchuan` mit Toleranz 0.001, nicht die im Entwurf zitierte `_accurate`-Variante mit 0.0001.)* Die Zellfortsätze haben im Bild eine messbare Breite — nur die SWC hat sie nicht gespeichert.

**Zweite Reihe, hier nur genannt:** DeepFlux (Vektorfeld auf 7-px-Band, `github.com/YukangWang/DeepFlux`), Topograph (breitenagnostischer Komponentengraph-Loss, `github.com/AlexanderHBerger/Topograph`), Sat2Graph (Knoten + Richtungsvektoren als Zusatzkanäle eines dichten Kopfes, `github.com/songtaohe/Sat2Graph`). Alle radiusfrei, alle mit Anpassungsaufwand.

**clDice als Metrik statt als Loss:** clDice misst genau das, was Dice verschweigt, und ist als Diagnosegröße uneingeschränkt brauchbar. Als Verlust auf einem 1-px-GT ist der Präzisionsterm `Tprec` problematisch — er verlangt, dass das Skelett der *Vorhersage* pixelgenau innerhalb des 1-px-GT liegt. Wenn clDice, dann mit kleinem α, oder nur die `Tsens`-Hälfte (was operativ Skeleton Recall Loss entspricht, nur teurer).

---

## Ehrliche Antwort auf den Einwand

**Wo der Nutzer recht hat.**

Das Feld arbeitet tatsächlich durchgängig mit SWC-Zentrallinien, und es gibt Verfahren, die ohne Radius auskommen — mehr als erwartet. 19 von 26 Arbeiten brauchen die Radiusspalte nicht, und mehrere ignorieren sie sogar dort, wo sie vorhanden wäre. Radius 0.000 ist kein Blocker. Es gibt auch, was er vermutet hat: mindestens ein Verfahren aus dem Neuronenfeld (NeuroFly), das produktiv, im Ganzhirnmaßstab und im Code verifiziert auf einem **1-Voxel-Label ohne jede Dilatation** trainiert. Sein Einwand war berechtigt und die Recherche hat ihn nicht widerlegt.

Er misst außerdem mit den Zusammenhangskomponenten etwas Vernünftiges und Diagnostisches — nur wird er dafür kaum Vergleichszahlen finden, weil das Feld diese Größe praktisch nie berichtet.

**Wo er nicht recht hat — und das ist die wichtigere Erkenntnis.**

Das Feld arbeitet mit SWC, aber es **trainiert fast nie auf dem rasterisierten SWC**. 23 von 26 Arbeiten verlassen das 1-px-Regime, bevor der Gradient fließt: pauschale Dilatation auf 2–7 px (Huang, Li, PointTree, Skeleton Recall, DeepFlux, DeepBranchTracer), weiche Felder mit Regressionsverlust (Mnih, Sironi, Oner, PointTree, Li, Xu), oder Aufgabe der dichten Segmentierung (DeepNeuron, NRTR, PointNeuron, Sat2Graph, Relationformer, Wolterink). Das 1-px-Skelett ist im Feld die **Annotationsform** und die **Ausgabeform** — dazwischen wird praktisch immer übersetzt. Die 1-px-Ausgabe entsteht am Ende durch Non-Maximum-Suppression, Flusssenken oder Skelettierung, nicht durch ein 1-px-Trainingsziel.

Drei Detailkorrekturen, die diese Bilanz noch verschärfen: DeepBranchTracers vermeintliches „1-px-Centerline-Label" hat r=1 als **Radius**, ist also eine 2–3 px breite Röhre mit 5–9× mehr Vordergrund. NeuroLinks „Ignore-Label" existiert nicht. Und 3D WaveUNet, das laut Rechercheentwurf die Labelkonstruktion nur schlecht dokumentiert, liest im Code sehr wohl die Radiusspalte — und wirft radiuslose Rekonstruktionen ausdrücklich aus dem Datensatz.

**Und die Antwort, die der Nutzer wahrscheinlich nicht erwartet hat:**

In **jeder einzelnen** der acht Neuronenarbeiten wird die Topologie **außerhalb** des Segmentierungsnetzes hergestellt. NeuroFly übersegmentiert absichtlich in Fragmente und verbindet sie in einer eigenen Path-Following-Stufe. DeepBranchTracer läuft mit Richtungs- und Radiusvorhersage iterativ die Faser entlang und überspringt Lücken. PointTree setzt ein Minimal-Information-Flow-Tree-Modell auf das Distanzfeld. DeepNeuron hat ein Siamese-Netz allein für die Frage, ob zwei Signalstellen zusammengehören. NRTR hat ein Konnektivitätsmodul. He et al. hängen einen Minimal-Cost-Path-Sucher an. **Niemand erwartet vom Segmentierungsnetz, dass seine Ausgabe eine Komponente ist.** Die Metrik „Zusammenhangskomponenten der Netzausgabe" misst im Feld nichts, was jemand optimiert — DIADEM, PyNeval und BigNeuron bewerten SWC-Graph gegen SWC-Graph, mit Distanz- und Topologiemaßen, nie mit Pixel-Overlap.

**Was das für die gemessenen Zahlen bedeutet.**

Dice 0,915 auf einem 1-px-Label ist ein deutlich stärkeres Ergebnis als derselbe Wert auf einer dicken Maske — dort halbiert schon eine Verschiebung um genau ein Pixel den Dice. Auf einem 1 px breiten Skelett ist außerdem **jedes Schaftpixel ein Artikulationspunkt**: ein einziges verfehltes Pixel spaltet die Komponente. Bei ~8,5 % Fehlanteil wären bei zufälliger Verteilung Größenordnungen mehr als 3,72 Komponenten zu erwarten. Dass es nur 3,72 sind, spricht dafür, dass der Fehler überwiegend als **seitlicher Versatz** auftritt (falsch positiv direkt neben falsch negativ, kein Schnitt) — exakt das Muster, das Oner et al. für handgetracte Zentrallinien beschreiben. Dice und Komponentenzahl sind also nicht widersprüchlich, sie messen zwei orthogonale Dinge, und Dice ist gegenüber Topologie blind.

Ein Verdacht lässt sich vorab ausschließen: SNTs Bresenham-Rasterisierung garantiert nur `diagonallyAdjacentOrEqual`, das Label ist also 8-verbunden. Da das GT in der Auswertung exakt 1,00 ergibt, passen Label und Metrik aber offensichtlich zusammen — ein 4-/8-Konnektivitätsartefakt ist damit **nicht** die Erklärung.

**Empfohlene Reihenfolge, aufsteigend nach Eingriffstiefe.**

1. **Diagnose (Stunden):** Wo liegen die Brüche — an Verzweigungen, bei schwachem Signal, an Kreuzungen? Komponentenzahl gegen Zelllänge auftragen. clDice als Metrik mitloggen.
2. **Architektur (ein Trainingslauf, keine Codeänderung):** Auflösungsstufen in `nnUNetPlans.json` von 6–7 auf 4 reduzieren. ThinCrack-Befund; das Problem tritt auf Trainingsdaten auf, das passt.
3. **Label (ein Trainingslauf):** Klasse 1 auf 3 px verdicken (radiusfrei, konform mit Huang 2 px / Li 3 Voxel / Skeleton Recall 2 px), 1-px-Ergebnis am Ende per Skelettierung zurückgewinnen. Damit verschwinden die Artikulationspunkte im Schaft vollständig. Erwartungsgemäß der größte Einzelhebel.
4. **Loss:** Skeleton Recall Loss (Trainer existiert, mit beiden Patches) oder CP-loss (~50 Zeilen, direkt auf das gemessene Symptom).
5. **Repräsentation (invasiv):** Distanzfeld + L1 nach Li/PointTree, mit vorgeschaltetem Resampling auf ≤1 px Knotenabstand.
6. **Topologie danach:** Wenn genau eine Komponente pro Zelle das Endprodukt sein muss, gehört ein Baum-/Tracer-Schritt hinter das Netz. Das Feld löst diese Anforderung nirgends im Segmentierungsverlust.
7. **Metrik:** Pixel-Dice nicht als Zielgröße verwenden. Das Feld evaluiert dünne Strukturen durchgängig mit Toleranz — Mnih ρ=3, Sironi ρ=2, ThinCrack d=5, Xu 3-px-Scheibe für TP/FP, DIADEM mit Schwellenzylindern.