# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Was das Projekt ist

Ein echtes Fliegen-Konnektom (MaleCNS-Datensatz, LIF-Neuronen) steuert eine
simulierte Drohne. Rein simulationsbasiert, kein Echtflug. Meilensteine
M0-M2 sind erledigt, M3 (Gain-Kalibrierung) ist der aktuelle Stand -- siehe
Status-Tabelle in `README.md` und den Phasenplan in `docs/projektplan.md`.

## Setup & Befehle

Verifiziert auf **Python 3.12** (macOS arm64, Homebrew). Die urspruengliche
3.10-Angabe in README/PATCHES.md stammt aus der Linux-Sandbox; 3.12 laeuft mit
den beiden PATCHES.md-Patches problemlos. 3.14 nicht nehmen -- pybullet,
opencv und `gym==0.26.2` haben dafuer noch keine verlaesslichen Wheels.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
bash scripts/setup_simulator.sh   # klont gym-pybullet-drones v1.0.0 + patcht
CFLAGS="-Dfdopen=fdopen" pip install pybullet   # siehe unten
pip install -r requirements.txt
```

**pybullet baut auf macOS nur mit `CFLAGS="-Dfdopen=fdopen"`.** pybullet hat
kein arm64-Wheel und kompiliert aus dem sdist; das mitgelieferte zlib
definiert `fdopen` als Makro, was die `fdopen`-Deklaration im aktuellen macOS-
SDK-Header zerlegt (`_stdio.h:322: expected identifier or '('`). Das
Vor-Definieren des Makros auf sich selbst laesst zlibs `#ifndef fdopen`
greifen. Einmal gebaut, liegt das Wheel im pip-Cache.

`simulator/` ist gitignored (vendorter Fremdcode) und wird nur durch
`setup_simulator.sh` erzeugt. Nie direkt editieren -- jede noetige Aenderung
gehoert als `sed`-Patch ins Setup-Skript **und** dokumentiert nach
`PATCHES.md`.

Alles, was den Simulator importiert, braucht `PYTHONPATH=simulator`:

```bash
PYTHONPATH=simulator python scripts/sanity_hover.py
PYTHONPATH=simulator python scripts/m2_real_connectome.py
```

Tests (brauchen `PYTHONPATH=.`, nicht den Simulator):

```bash
PYTHONPATH=. python -m pytest connectome/tests/
PYTHONPATH=. python -m pytest connectome/tests/test_real_connectome.py::test_symmetric_motion_produces_balanced_output
```

`connectome/tests/test_optomotor.py` haelt die Richtungsselektivitaet fest und
laeuft bewusst ohne Simulator (synthetische driftende Balken statt PyBullet).
Achtung beim Erweitern: ein perfekt periodisches Rechteckgitter mit identischen
Zeilen liefert Farneback exakt 0.0 Fluss -- der Reiz braucht unregelmaessige
Balkenbreiten und Rauschen, und das Rauschen muss mit dem Muster mitwandern
(statisches Pixelrauschen verankert die Flussschaetzung bei null).

Jede Testdatei ist auch direkt ausfuehrbar (`python connectome/tests/test_toy_network.py`).
`test_real_connectome.py` skippt sich selbst, wenn `data/raw/` fehlt.
Die volle Suite braucht ~1 Minute (das echte Konnektom wird pro Prozess einmal
geladen und gecached).

Kein Linter, kein Formatter, keine CI konfiguriert.

## Architektur

Datenfluss (eine Iteration pro Kamerabild, 30 Hz):

```
env._getDroneImages() -> medulla_encoder -> LIFNetwork (~333 Steps a 0.1ms)
  -> reset_spike_window() -> motor_decoder -> (thrust, yaw)
  -> Sollwert-Nudge -> DSLPIDControl -> env.step({0: rpm})
```

Wichtig: Der **PID-Regler macht die Fluglage** (Roll/Pitch/Position). Das
Konnektom liefert nur zwei Kanaele -- Gier (links-rechts-Differenz) und
Hoehe (Summe) -- als Sollwert-Drift obendrauf. Das ist Phase A aus dem
Projektplan, kein Versehen; Phase B (4 DOF) ist noch offen.

### Die Vertrags-Schnittstelle zwischen den Modulen

Alle Netzwerk-Loader (`data_loader.make_toy_network`,
`data_loader.fetch_from_neuprint`, `local_maleCNS.local_fetch_subnetwork`,
`local_maleCNS.local_fetch_lr_subnetwork`) geben **dasselbe dict** zurueck:

```python
{"n_neurons": int,
 "weights": scipy.sparse,          # weights[post, pre]
 "cell_type_indices": dict[str, np.ndarray]}   # Typname -> Neuron-Indizes
```

Encoder und Decoder sprechen ausschliesslich ueber `cell_type_indices` mit
dem Netzwerk. Deshalb ist "synthetisch vs. echt" am Call-Site eine Zeile,
kein Rewrite. Neue Datenquellen muessen diese Form einhalten.

### Zwei parallele Pfade, nicht einer

| | synthetisch | echtes Konnektom |
|---|---|---|
| Loader | `make_toy_network()` | `local_fetch_lr_subnetwork()` |
| Encoder | `encode_to_drive()` (Richtungs-/Looming-Split) | `encode_to_drive_hemifield()` (L/R-Bildhaelfte -> `somaSide`) |
| Pools | `T4_*`/`T5_*`/`LC4`/`LPLC2`, `motor_left/right` | `visual_L/R` (T4/T5), `motor_left/right` (HS/H1/H2) |
| Skript | `scripts/m2_closed_loop.py` | `scripts/m2_real_connectome.py` |
| Test | `test_toy_network.py` | `test_real_connectome.py` |

Der synthetische Pfad bleibt absichtlich bestehen: er ist der
Verdrahtungs-Regressionstest, der ohne den 1,1-GB-Download laeuft. Aenderungen
an Encoder/Netzwerk/Decoder in beiden Pfaden pruefen.

### Die Welt muss texturiert sein, sonst ist der Regelkreis tot

`VisionAviary`s Default-Welt ist eine leere Bodenebene. Der optische Fluss ist
darin ~0, also bekommen T4/T5 keinen Drive und die Konnektom-Ausgabe faellt auf
exakt 0.000, sobald die Drohne nicht mehr steigt -- der PID haelt sie oben, die
Fliege traegt nichts bei. `world/room.py` baut deshalb einen Streifenraum
(gemessen: 0% -> 43% aktive Frames im Steady State, `scripts/m2_room.py`).

Zwei Render-Eigenheiten, beide gemessen, nicht geraten:

- **Streifen als Geometrie, nicht als Textur.** PyBullets GEOM_BOX-UV-Mapping
  verzerrt eine Textur auf duennen Platten; vertikale Balken kamen als
  horizontale Schlieren raus. Separate Box-Bodies umgehen das UV-Mapping.
- **Keine Farbwerte nahe Schwarz.** Eine Innenwand rendert mit ~0.69 der vollen
  Helligkeit (weisse Wand: mean 176/255 aus Drohnensicht). Ein 0.08/0.92-Paar
  landet bei 14/162, die dunklen Balken saufen ab. `BAR_DARK`/`BAR_LIGHT` in
  `world/room.py` halten deshalb Abstand zu beiden Enden.
- Die Lichtrichtung ist nicht aenderbar: `_getDroneImages()` hardcodet die
  `getCameraImage()`-Lichtparameter, und `configureDebugVisualizer(lightPosition=)`
  wirkt nur auf die OpenGL-GUI, nicht auf den headless genutzten TinyRenderer.

`build_room()` muss **nach** `env.reset()` laufen -- `reset()` ruft
`p.resetSimulation()` und wuerde den Raum wieder loeschen.

### Richtung geht im Encoder verloren, wenn man den falschen nimmt

`encode_to_drive_hemifield()` speist `np.abs(flow)` pro Bildhälfte ein und wirft
damit das Vorzeichen der Bewegung weg. Gemessen an der rotierenden Trommel: der
vorzeichenbehaftete Fluss schlägt sauber von −1.14/−1.02 (CCW) auf +0.99/+1.16
(CW) um, das benutzte `|flow|` kaum (0.94/0.69 gegen 0.66/1.06). Ergebnis: keine
Richtungsselektivität, egal wie man den Gain waehlt.

`encode_to_drive_progressive()` behebt das und behaelt die anatomische
Zuordnung: HS-Zellen einer Seite sind auf **progressive** (front-to-back)
Bewegung des eigenen Auges selektiv, also `visual_L` <- Bewegung nach links in
der linken Bildhaelfte, `visual_R` <- nach rechts in der rechten. Damit feuert
pro Drehrichtung genau ein Pool, der andere schweigt.

Fuer neue Reize immer erst `scripts/m3_gain_sweep.py` laufen lassen, bevor man
Verhalten interpretiert -- ein zu hoher Gain sieht nicht wie ein Fehler aus,
sondern wie ein Null-Ergebnis.

### Jede Schicht braucht ihren eigenen Gewichts-Massstab

Das Netz hat drei Schichten mit zwei Groessenordnungen Fan-in-Unterschied: eine
HS-Zelle integriert ~6800 T4/T5-Zellen, ein absteigendes Neuron nur ~12
HS-Zellen. Ein einziger globaler `WEIGHT_SCALE` kann beide nicht bedienen --
mit dem fuer HS kalibrierten Wert erreichen die DNs rechnerisch ~0.1 gegen eine
Schwelle von 1.0 und feuern bei *keinem* Encoder-Gain (gemessen: stumm von 1.0
bis 16.0). `lif_network.scale_incoming()` skaliert deshalb die Eingaenge einer
Schicht separat; `DN_SCALE = 12.0` ist der kalibrierte Wert.

Das ist eine Modellannahme, keine Aussage des Konnektoms -- die Daten liefern
Konnektivitaet, keine Biophysik. Entsprechend benennen, nicht verstecken.

Ein zu niedriger Massstab sieht aus wie ein kaputter Pfad, ein zu hoher wie
Saettigung -- beide liefern ein sauberes Null-Ergebnis. `test_optomotor.py`
haelt den Wert deshalb von beiden Seiten fest.

### Drei Encoder, und der Unterschied ist nicht kosmetisch

- `encode_to_drive_hemifield` -- `|flow|`, wirft die Richtung weg. Nur fuer den
  M2-Pfad und die alten Tests da.
- `encode_to_drive_progressive` -- Farneback-Fluss, richtungsselektiv, misst
  Geschwindigkeit.
- `encode_to_drive_reichardt` -- Korrelations-Detektor. Sein Optimum liegt bei
  fester Temporalfrequenz (0.25 Zyklen/Frame bei einem Frame Verzoegerung),
  unabhaengig von der Streifenbreite -- die Fliegen-Signatur, die ein
  Geschwindigkeitsschaetzer prinzipiell nicht haben kann. Gains sind pro
  Encoder verschieden: progressive 1.0-4.0, reichardt ~10 (seine Ausgabe ist
  eine Groessenordnung kleiner, weil sie ein Produkt zweier Kontraste ist).

Praktische Folgen beim Interpretieren von Ergebnissen:
- Reize oberhalb von ~5 px Verschiebung pro Bild verlassen den Bereich, in dem
  Farneback (winsize=9) zuverlaessig schaetzt; die Antwort faellt dann ab.
- Laeuft ein periodisches Muster mehr als einen halben Balken pro Bild weiter,
  **kippt das Vorzeichen** (gemessen bei 24 Streifen / 4.8 rad/s). Das sieht wie
  eine Reaktion in die Gegenrichtung aus und ist reines Sampling-Artefakt.
- Tuning-Aussagen brauchen einen Reiz mit **einer** Ortsfrequenz. Die
  rotierende Trommel hat keine: ihre Balkenperiode in Pixeln variiert ueber das
  Blickfeld, und der Mosaikboden ist breitbandig und steht still. Der
  Trommel-Sweep kam deshalb flach und nicht entscheidbar heraus; die belastbare
  Messung steht in `test_optomotor.py` gegen ein Sinusgitter.
- Ein argmax auf einer flachen Kurve ist kein Maximum. Erst die Flachheit
  pruefen (`m3_tuning_curve.py` gibt sie aus), dann interpretieren -- argmax
  und Schwerpunkt wanderten hier um mehr als Faktor zwei auseinander.
- Die DN-Schicht hat ein enges Fenster: bei Gain 1.0 antwortet sie nur auf 2 von
  6 Geschwindigkeiten, bei Gain 4.0 auf 6 von 6. Ein stummer Messpunkt heisst
  also nicht "kein Reiz", sondern oft "unter der Feuerschwelle" -- vor jeder
  Aussage ueber Tuning den Encoder-Drive getrennt mitmessen.

### Zelltypnamen in MaleCNS vor dem Aufgeben pruefen

M4 galt eine Zeit lang als blockiert, weil `VS\d+` nichts fand. MaleCNS fasst
alle acht Subtypen des vertikalen Systems unter dem Typnamen **`VS`** zusammen
-- die Spalte `flywireType` schreibt es als `VS1,...,VS8` aus. Die Annotationen
haben mehrere Namensfelder (`type`, `hemibrainType`, `flywireType`, `synonyms`,
`supertype`); bei einer Fehlanzeige alle durchsuchen, bevor man einen Zelltyp
fuer abwesend erklaert.

### Die Drohnenkamera rollt erst seit Patch #3 mit

`cameraUpVector` war in `_getDroneImages()` hart auf die Welt-Hochachse
gesetzt, Roll war fuer die Kamera unsichtbar (gemessen: 46 Grad Roll aenderten
das Bild um 2.9 Graustufen, ein Nicken um 44). Das ist die einzige
Verhaltensaenderung am vendorten Simulator, im Unterschied zu den reinen
Kompatibilitaets-Patches #1 und #2 -- siehe PATCHES.md.

### Erst mitteln, dann rektifizieren

Der teuerste Fehler dieses Projekts, in vier Varianten wiederholt: `clip()` vor
`mean()`. Bei jeder Opponenz-Rechnung -- Links gegen Rechts, Auswaerts gegen
Einwaerts -- zerstoert Rektifizierung pro Pixel die Aufhebung, auf der die
Unterscheidung beruht. Der Looming-Detektor feuerte dadurch beim Drehen
staerker als beim drohenden Aufprall (0.48x-0.65x), und keine Menge
Retinotopie oder Zusatzkanaele hat das repariert. Erst das vorzeichenbehaftete
Feld ueber die Bildmitte mitteln, dann rektifizieren -- damit liegt die
Drehung bei exakt 0.

Wer hier etwas aendert: `connectome/tests/test_looming.py` haelt das fest, und
die Fehlermeldung nennt die Ursache beim Namen.

### Kontrollbedingungen muessen die Textur teilen

Eine Zwischenversion sah mit 4.39x nach der Loesung aus und war konfundiert:
Drehkontrolle auf der gestreiften Trommel, Anflugobjekt mosaikiert. Ein
vertikales Gitter hat keinen vertikalen Helligkeitsgradienten, der damals
verwendete Vertikalkanal las darauf exakt 0.00000 -- bei Drehung wie bei
Expansion. Der Vergleich trennte zwei Texturen, nicht zwei Bewegungen.

Deshalb gibt es in `m5_looming.py` die self-yaw-Kontrolle: die Drohne dreht
sich selbst, wodurch dieselbe Szene schwenkt, die beim Anflug stillsteht.

### Kalibrierung ist der empfindliche Teil

- `GAIN` skaliert den **Encoder-Input**. 8.0 stammt vom Spielzeugnetz und gilt
  nur fuer den statischen Reiz; am Trommelreiz saettigt es beide Motor-Pools auf
  den Decoder-Deckel (gemessen: thrust 0.49898 = 0.25+0.25), womit
  `yaw = links - rechts` per Konstruktion null wird. Kalibrierte Werte:
  `hemifield` 0.7, `progressive` 1.0 (`GAIN_BY_ENCODER` in
  `scripts/m3_optomotor.py`).
- `WEIGHT_SCALE = 0.0015` skaliert die **echten Synapsen-Gewichte** herunter
  (Rohdaten sind Synapsenzahlen, Mittel ~2,6, kumuliert 10.000+ pro
  Motor-Neuron). Ohne diese Skalierung saettigen alle Motor-Neuronen auf den
  Decoder-Deckel 0,2500 und der Gier-Kanal wird blind. Begruendung im
  Docstring von `connectome/tests/test_real_connectome.py`, Sweep-Historie in
  `docs/connectome-data-access.md`.
- Beide Konstanten sind **dupliziert** in `scripts/m2_real_connectome.py` und
  `connectome/tests/test_real_connectome.py` -- beim Aendern beide anfassen.

### Bekannte Fallstricke (teuer erkauft, nicht neu herausfinden)

- `local_maleCNS.py` streamt die 1,1-GB-Gewichtsdatei batchweise per
  `pyarrow.ipc`. Ein `pd.read_feather()` auf diese Datei OOM-killt den
  Prozess. Nicht "vereinfachen".
- `neuprint-python` haelt den Default-Client nur per **weakref** -- eine
  `Client(...)`-Expression ohne starke Referenz wird sofort weggeraeumt.
  Deshalb wird der Client in `data_loader.py` explizit als `client=` an jeden
  Query durchgereicht.
- `neuprint.janelia.org` hostet mehrere Datensaetze und akzeptiert
  `dataset=None` nicht; Default ist `male-cns:v1.0`.
- Der LIF-Kern braucht den exponentiell gefilterten Synapsenstrom
  (`tau_syn`), sonst erreichen seltene Spikes nie die Schwelle.
- `LIFNetwork` ist bewusst numpy/scipy-only und **nicht** auf 166.700 Neuronen
  ausgelegt. Fuer den vollen Massstab waere ein GeNN-/Numba-Backend mit
  gleicher `__init__`/`step()`-Signatur der Austauschpunkt.

## Konventionen

- Code und Code-Kommentare **englisch**, Dokumentation, Commit-Messages und
  Nutzertexte **deutsch**. Commit-Messages ohne Umlaute (ae/oe/ue/ss).
- Skripte in `scripts/` sind Ein-Zweck-Top-Level-Skripte ohne `main()`, die
  ihre `sys.path`-Eintraege selbst setzen und Ergebnisse als Klartext-Tabelle
  plus explizites Pass/Fail-Urteil ausgeben. Diesem Muster folgen.
- Nicht-triviale Entscheidungen werden im Modul-Docstring begruendet
  (inklusive verworfener Alternativen). Diese Docstrings sind hier die
  primaere Dokumentation -- bei Aenderungen mitpflegen.
- `data/`, `files/`, `simulator/`, `.env*` (ausser `.env.example`) sind
  gitignored.
