# fly-drone-sim

Ein echtes Fliegen-Konnektom (166.700 LIF-Neuronen, MaleCNS-Datensatz)
steuert eine simulierte Drohne statt eines physischen Roboters. Basiert auf
dem [boat.horse/fly](https://boat.horse/fly/)-Projekt (Fly Rig Anatomy),
das dieselbe Konnektom-Pipeline aktuell an einen Strandbeest-Laufroboter
anschließt.

Vollständig simulationsbasiert -- kein Echtflug geplant.

## Inhalt

- [Überblick](#überblick)
- [Status](#status)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Nutzung](#nutzung)
- [Echte Konnektom-Daten](#echte-konnektom-daten)
- [Projektstruktur](#projektstruktur)
- [Design-Entscheidungen](#design-entscheidungen)
- [Datenquellen & Lizenzen](#datenquellen--lizenzen)
- [Weiterführende Dokumentation](#weiterführende-dokumentation)

## Überblick

```
Drohnen-Kamera (RGB) -> Medulla-Encoder (optischer Fluss)
   -> LIF-Konnektom-Sim (T4/T5 -> HS/H1/H2) -> Motor-Decoder
   -> Gier-/Höhen-Sollwert -> Flugregler (PID) -> Drohnen-Physik
```

Roll/Pitch/Position übernimmt ein fertiger, bewährter PID-Regler
(`DSLPIDControl` aus gym-pybullet-drones); nur Gier und Höhe kommen aus der
Konnektom-Simulation. Details und Architekturdiagramm: [`docs/projektplan.md`](docs/projektplan.md).

## Status

| Meilenstein | Beschreibung | Stand |
|---|---|---|
| M0 | Simulator gewählt, RPM-Ansteuerung + Kamera-Sanity-Check | ✅ erledigt |
| M1 | Medulla-Encoder -> LIF-Netzwerk -> Motor-Decoder, offline getestet | ✅ erledigt (synthetisch + echtes Konnektom) |
| M2 | Geschlossener Regelkreis, Phase A (Gier + Höhe) | ✅ erledigt (synthetisch + echtes Konnektom) |
| M3 | Feintuning (Gain-Kalibrierung Spike-Rate <-> RPM-Offset) | ✅ erledigt (Gain pro Encoder kalibriert, Optomotorik-Reaktion nachgewiesen) |
| M3b | Absteigende Neuronen als echte Ausgabeschicht | ✅ erledigt (T4/T5 → HS/H1/H2 → DNb03/DNg41/DNp15/DNp17/DNa02) |
| M3c | Hassenstein-Reichardt-Korrelator statt optischem Fluss | ✅ erledigt (Temporalfrequenz-Tuning + Reverse-Phi nachgewiesen) |
| M4 | Phase B, 4 Freiheitsgrade | ⚠️ Pfad + Auslesetest erledigt; im Regelkreis trägt Roll **nicht** bei (Latenz, s. u.) |
| M5 | Auswertung (Stretch): emergentes Looming-Ausweichen / Höhenhaltung | ✅ erledigt (Winkelgrößen-Schwelle + Spezifität gegen Drehung, drei Kontrollen bei exakt 0) |

**Aktuelles Ergebnis (M2, `scripts/m2_real_connectome.py`):** Die Drohne
hält mit dem echten MaleCNS-Konnektom (T4/T5 -> HS/H1/H2, 13.597 Neuronen,
207.059 Synapsen -- nicht dem synthetischen Testnetz) stabil die Höhe und
reagiert im Gier-Kanal sichtbar auf das, was die Kamera an Bewegung sieht.
Details, inklusive der Gewichts-Kalibrierung, die dafür nötig war:
[`docs/connectome-data-access.md`](docs/connectome-data-access.md).

**Optomotorik-Nachweis (M3, `scripts/m3_optomotor.py`):** In einer rotierenden
Streifentrommel dreht die Drohne mit der Trommel mit und kehrt die Drehrichtung
um, wenn die Trommel es tut — die klassische Optomotorik-Reaktion der Fliege,
getrieben von genau dem T4/T5 → HS/H1/H2-Pfad. Angebunden (Pose fixiert, reiner
Auslesetest): CCW +0.075 ± 0.003 gegen CW −0.080 ± 0.005 über je 3 Durchgänge,
vollständig überlappungsfrei. Aufzeichnung des Laufs als MP4 und Zeitreihen-Plot
unter `files/optomotor/`.

Seit M3b wird nicht mehr aus HS/H1/H2 dekodiert, sondern aus den **absteigenden
Neuronen**, die diese treiben — dem tatsächlichen Ausgang des Gehirns zum
Bauchmark statt aus Interneuronen. Die Reaktion bleibt richtungsselektiv
(CCW −0.0118 ± 0.0007 gegen CW +0.0140 ± 0.0009). Welche DNs das Lenksignal
tragen, wurde gemessen statt geraten: **DNg41** ist mit Abstand am stärksten
(−34 / +31 Hz Seitendifferenz), gefolgt von DNp15 und DNb03. **DNa02 — das
kanonische Lenk-DN der Literatur — bleibt in diesem Modell stumm**, weil es nur
Gewicht 182 aus HS bekommt gegen DNg41s 1166; sein übriger Eingang liegt
außerhalb dieses Teilnetzes.

**Bewegungsdetektion (M3c):** Die Richtungsselektivität entstand ursprünglich
gar nicht im Konnektom, sondern in `medulla_encoder` durch Farnebacks optischen
Fluss — ein Algorithmus an genau der Stelle, für die T4/T5 berühmt sind.
`encode_to_drive_reichardt()` ersetzt ihn durch einen
Hassenstein-Reichardt-Korrelator: ein verzögertes Signal wird mit dem
unverzögerten des Nachbarpunkts multipliziert, die spiegelbildliche Paarung
abgezogen. Für ein driftendes Gitter ergibt das `resp ∝ sin(k·s)·sin(k·v·Δ)`,
das Maximum liegt also bei fester *Temporalfrequenz* unabhängig von der
Streifenbreite — die Signatur echter Fliegen (Götz), die ein
Geschwindigkeitsschätzer prinzipiell nicht haben kann.

Kontrolliert nachgewiesen (`connectome/tests/test_optomotor.py`): bei Perioden
von 16 und 32 px liegt das Optimum in beiden Fällen bei **0.25 Zyklen/Frame**,
also bei doppelter Driftgeschwindigkeit für die doppelte Periode. Oberhalb von
0.5 Zyklen/Frame kehrt sich das Vorzeichen um — Reverse-Phi, das Fliegen
ebenfalls zeigen.

Am rotierenden Trommelreiz (`scripts/m3_tuning_curve.py`) ist die Frage
dagegen **nicht entscheidbar**: die Antwortkurven sind flach, argmax und
Schwerpunkt widersprechen sich, und eine Trommel ist keine einzelne
Ortsfrequenz — die Balkenperiode in Pixeln variiert über das Blickfeld und der
Mosaikboden ist breitbandig und steht still. Was der Sweep klar zeigt: der
Korrelator liefert einen glatten, monotonen Drive bis zu seinem Optimum, wo
Farneback oberhalb von ~5 px Verschiebung pro Bild erratisch wird und
schließlich kippt.

Nebenbefund: bei 24 Streifen und 4.8 rad/s **kehrt sich das Vorzeichen um** —
die Trommel läuft dann mehr als einen halben Balken pro Bild weiter, klassisches
Bewegungs-Aliasing. Das Skript markiert diesen Punkt vorab als `ALIASED`, damit
er nicht als Abstimmung missgedeutet wird.

Nötig dafür war ein richtungsselektiver Encoder
(`encode_to_drive_progressive`); mit dem älteren `encode_to_drive_hemifield`,
der `|flow|` benutzt und damit das Vorzeichen der Bewegung wegwirft, gibt es
keine messbare Reaktion — beide Varianten stehen als Vergleich im Skript.

**Looming-Pfad (M5, `scripts/m5_looming.py`):** LC4/LPLC2 → Fluchtneuronen
(DNp01/Giant Fiber, DNp04 u. a.) ist im Konnektom kräftiger verdrahtet als der
Kurskontrollpfad — Gewicht 14995 gegen 1266. Beim Anflug feuert die Schicht
einen **Transienten** (5 von 62 Bildern), und beide Anfluggeschwindigkeiten
peaken bei derselben **Winkelgröße** (34.7° bzw. 33.4°), nicht bei derselben
Zeit oder Distanz — die klassische Beschreibung der Giant-Fiber-Auslösung, hier
ohne Tuning aus Detektor und Geometrie entstanden.

**Spezifität, fünfter Anlauf.** Verhältnis Anflug zu Drehung; unter 1.0 heißt,
der Fluchtkreis feuert beim Drehen stärker als beim drohenden Aufprall:

| Encoder-Variante | Selektivität |
|---|---|
| Auswärtsbewegung pro Bildhälfte | 0.52x |
| deren räumliche Divergenz | 0.48x |
| dieselbe auf retinotopem 4×8-Raster | 0.65x |
| horizontal + vertikal konjunktiv | ~~4.39x~~ konfundiert, real 1.5x |
| **Opponenz: erst mitteln, dann rektifizieren** | **Drehung exakt 0** |

Alle vier Fehlschläge hatten dieselbe Ursache: **Rektifizierung pro Pixel vor
dem Mitteln**. Das wirft die negative Hälfte des Radialfelds weg — und genau
die hebt eine Drehung auf. Erst die vorzeichenbehaftete Radialkomponente über
die Bildmitte mitteln, dann rektifizieren.

Drei Kontrollen, alle bei 0.00000: Rückzug, rotierende Trommel, und
Eigendrehung der Drohne (letztere schwenkt die ganze Szene inklusive des
isotropen Mosaikbodens — die Trommelkontrolle allein würde nicht reichen, weil
ein vertikales Gitter ein Reiz ist, auf dem man leicht aus den falschen Gründen
selektiv aussieht).

**Vertikales System (M4, `scripts/m4_vertical.py`):** Dieselbe Architektur eine
Achse weiter — T4/T5 → **VS** → absteigende Neuronen (DNp20 u. a.), gespeist
mit vertikaler statt horizontaler Bildbewegung. Ein neuer Decoder war nicht
nötig: Rollen dreht die beiden Bildhälften gegensinnig, Nicken gleichsinnig,
also trennen dieselbe Differenz und Summe die Kanäle, die auf der horizontalen
Seite Gier und Schub tragen.

Roll kehrt sauber um (+0.0122 gegen −0.0052 über je 3 Durchgänge,
überlappungsfrei), und Gieren leckt **exakt 0** in den Roll-Kanal. Nicken
landet wie erwartet im Summenkanal, aber nur in **einer** Richtung: jede
Population ist rektifiziert und meldet nur Abwärtsbewegung, und `somaSide`
liefert nur zwei Pools.

Zwei Korrekturen waren dafür nötig:

- **M4 war nie an fehlenden Daten blockiert, sondern an einer Regex.** MaleCNS
  fasst alle acht VS-Subtypen unter dem einzigen Typnamen `VS` zusammen
  (`flywireType` schreibt es aus: `VS1,…,VS8`); die frühere Suche nach `VS\d+`
  verlangte eine Ziffer. 18 Zellen, 9 pro Seite, mit Gewicht 158.026 aus T4/T5
  — 8779 pro VS-Zelle gegen 13.164 pro HS-Zelle.
- **Die Drohnenkamera rollte nicht mit der Drohne.** `cameraUpVector` war im
  vendorten Simulator hart auf die Welt-Hochachse gesetzt; ein Roll um 46°
  änderte das Bild um 2.9 Graustufen, also gar nicht. Behoben als Patch #3,
  siehe [`PATCHES.md`](PATCHES.md).

**Phase B im Regelkreis (`scripts/m4_closed_loop.py`):** Beide Teilnetze laufen
zusammen — 27.244 Neuronen, 30 Hz, auf einer fliegenden Drohne. Das Konnektom
liefert Sollwerte auf allen vier Kanälen, die Drohne bleibt oben.

**Der Roll-Kanal trägt im Flug aber nichts bei.** Bei kalibriertem Gain feuert
er nie; auf das 20-fache gezwungen erreicht er eine Korrelation von 0.21 mit
der tatsächlichen Drehrate. Die Ursache ist nicht das Konnektom, sondern die
Taktrate: `DSLPIDControl` drückt die Störung auf 0.110 rad und erstickt den
Transienten in ~0.2 s — etwa **sechs Kamerabilder**. Ein 30-Hz-Bildtakt mit
20-ms-Decodierfenster (zusammen ~53 ms Latenz) hat auf dieser Zeitskala keine
Auflösung mehr. Im Auslesetest, wo die Drehung 1.2 s lang anliegt, liefert
derselbe Pfad ein sauberes umkehrendes Signal.

Den PD-Regler *ersetzen* — die andere Lesart von Phase B — ist damit ebenfalls
nicht erreichbar, und das ist gemessen statt vermutet: senkt man
`D_COEFF_TOR[0]`, bleibt die Rollachse bis 11000 praktisch unbewegt (0.163 rad)
und überschlägt sich ab 10000. Der Übergang ist eine Bifurkation, kein Verlauf
— es gibt kein Regime, in dem die Achse lose genug für ein visuelles Signal und
zugleich flugfähig ist.

**Freier Flug im Labyrinth (`scripts/m6_maze.py`):** Die Drohne fliegt
selbstständig durch ein Labyrinth (`world.build_maze`, als ASCII-Karte
definiert) und **nimmt die erste Kurve**, wo sie ungesteuert stur in die Wand
fliegt. Null Wandkontakte über 40 s gegen 891 ohne Steuerung; mit umgekehrtem
Vorzeichen zieht sie nach 1.8 m hinein.

Welcher Auslesekanal das leistet, wurde gemessen, und der naheliegende verlor.
Drive gegen Wandabstand im Korridorflug:

| Auslesung | 8–12 m | 0.6–1 m |
|---|---|---|
| global radial (Looming) | 0.0014 | 0.0039, Maximum bei 10.8 m |
| nur frontales Drittel | 0.0001 | **exakt 0** ab 1.5 m |
| **seitliches Gleichgewicht** | 0.0036 | **0.0104**, monoton |

Der Grund ist geometrisch: Bei Vorwärtsflug liegt die Wand, auf die man
zufliegt, im **Expansionsfokus** — dort ist die Bildbewegung null — während die
Seitenwände bei konstantem Abstand und hoher Winkelgeschwindigkeit jede globale
Mittelung dominieren. Ein Looming-Detektor meldet ein Objekt, das sich einem
ruhenden Beobachter nähert, gut; eine Wand, auf die man zufliegt, schlecht.

`encode_to_drive_centring` nutzt deshalb den Betrag pro Bildhälfte — ausgerechnet
die Rechnung, die für die Optomotorik falsch war, weil sie die Richtung
wegwirft. Hier ist das richtig: beide Wände streichen rückwärts, nur ihr
Tempoverhältnis zählt.

## Voraussetzungen

- Python 3.10 bis 3.12 (siehe [Design-Entscheidungen](#design-entscheidungen));
  verifiziert auf 3.12. Nicht 3.13+ -- dafür fehlen pybullet/opencv/`gym`-Wheels.
- ~30 MB Platz für den vendorten Simulator, optional ~1.1 GB für die echten
  Konnektom-Rohdaten

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
bash scripts/setup_simulator.sh   # klont + patcht simulator/ (gitignored, ~28MB)
pip install -r requirements.txt
```

**Auf macOS** hat pybullet kein fertiges Wheel und muss kompiliert werden; das
schlägt mit dem aktuellen SDK fehl (`_stdio.h: expected identifier or '('`),
weil das mitgelieferte zlib `fdopen` als Makro definiert. Deshalb pybullet dort
vorab einzeln installieren:

```bash
CFLAGS="-Dfdopen=fdopen" pip install pybullet
```

Sanity-Check:

```bash
PYTHONPATH=simulator python scripts/sanity_hover.py
```

Erwartete Ausgabe: `HOVER_RPM`/`MAX_RPM`-Werte, eine Positionsangabe nach 3s
simulierter Zeit, und `files/sanity/frame_0.png` (ein 48x64-Kamerabild aus
Drohnensicht).

## Nutzung

| Skript | Was es macht | Konnektom |
|---|---|---|
| `scripts/sanity_hover.py` | M0: RPM-Ansteuerung + Kamerabild, kein Konnektom | -- |
| `scripts/capture_frame_sequence.py` | nimmt die Kamera-Sequenz für M1 auf (`files/m1/frame_sequence.npy`) | -- |
| `scripts/m1_offline_test.py` | M1: Kamera-Sequenz offline durch die Pipeline schicken (braucht die Aufnahme oben) | synthetisch |
| `scripts/m2_closed_loop.py` | M2: geschlossener Regelkreis | synthetisch |
| `scripts/m2_real_connectome.py` | M2: geschlossener Regelkreis | **echtes MaleCNS-Konnektom** (braucht die Rohdaten, siehe unten) |
| `scripts/m2_room.py` | M2 im texturierten Raum, direkt gegen die leere Welt gemessen | **echtes MaleCNS-Konnektom** |
| `scripts/preview_room.py` | rendert den Raum von außen + aus Drohnensicht nach `files/room/` | -- |
| `scripts/m3_optomotor.py` | Optomotorik-Experiment: rotierende Streifentrommel, beide Encoder im Vergleich; schreibt `files/optomotor/flight.mp4` + `response.png` | **echtes MaleCNS-Konnektom** |
| `scripts/m3_gain_sweep.py` | Gain-Kalibrierung gegen den Trommelreiz (`SWEEP_GAINS=`, `M3_ENCODER=`) | **echtes MaleCNS-Konnektom** |
| `scripts/m3_tuning_curve.py` | Tuningkurve: Geschwindigkeit vs. Temporalfrequenz, zwei Streifenzahlen | **echtes MaleCNS-Konnektom** |

Alle Skripte mit `PYTHONPATH=simulator` ausführen, z. B.:

```bash
PYTHONPATH=simulator python scripts/m2_real_connectome.py
```

Tests:

```bash
PYTHONPATH=. python -m pytest connectome/tests/
```

(`test_real_connectome.py` wird automatisch übersprungen, wenn die lokalen
Rohdaten -- siehe unten -- noch nicht heruntergeladen sind.)

## Echte Konnektom-Daten

Zwei Wege, echte MaleCNS-Daten zu benutzen:

### a) Lokale Bulk-Dateien (empfohlen, kein Account nötig)

Direkter, öffentlicher Download (Janelia FlyEM + Google Research,
CC-BY-lizenziert) -- kein neuPrint-Account/Token nötig, keine
Query-Größenlimits. Zwei Dateien: Body-Annotationen (~13 MB) und die
komplette Konnektivitäts-Gewichtsmatrix (~1.1 GB, ~152 Mio. Zeilen).

```bash
mkdir -p data/raw
curl -L -o data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather
curl -L -o data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather
```

Unter Windows/PowerShell: `curl.exe` statt `curl` verwenden (sonst greift
der `Invoke-WebRequest`-Alias, dessen Fortschrittsbalken große Downloads
stark ausbremst) und vorher `$ProgressPreference = 'SilentlyContinue'`
setzen.

Danach direkt nutzbar, ganz ohne Netzwerkzugriff:

```bash
python -c "from connectome.local_maleCNS import local_fetch_lr_subnetwork; n = local_fetch_lr_subnetwork(); print(n['n_neurons'], 'Neuronen,', n['weights'].nnz, 'Synapsen')"
PYTHONPATH=simulator python scripts/m2_real_connectome.py
```

Alle offiziellen Download-Links: https://male-cns.janelia.org/download/.
`connectome/local_maleCNS.py` liest die 1.1-GB-Gewichtsdatei
speicherschonend im Batch-Streaming-Verfahren -- Details im Docstring dort.

### b) neuPrint-API (für einzelne Ad-hoc-Queries)

`.env.example` nach `.env` kopieren und den neuPrint-API-Token eintragen
(`.env` ist gitignored). `connectome/data_loader.py` liest ihn automatisch
ein. Details: [`docs/connectome-data-access.md`](docs/connectome-data-access.md).

## Projektstruktur

```
flydrone/
  simulator/     vendorter gym-pybullet-drones v1.0.0, gitignored (siehe PATCHES.md)
  scripts/       ausführbare Ein-Zweck-Skripte (siehe Nutzung oben)
  world/
    room.py        texturierter 3D-Raum + rotierbare Optomotorik-Trommel
  connectome/
    medulla_encoder.py   Kamerabild -> optischer Fluss -> Drive-Vektor
    lif_network.py       LIF-Netzwerk-Engine (scipy.sparse)
    motor_decoder.py     Spike-Raten -> normierte Aktuator-Werte
    data_loader.py        synthetisches Testnetz + neuPrint-API-Anbindung
    local_maleCNS.py      echte MaleCNS-Daten aus den lokalen Bulk-Dateien
    tests/                 Regressionstests (synthetisch + echtes Konnektom)
  data/raw/      lokale MaleCNS-Bulk-Dateien, gitignored (siehe oben)
  files/         Simulator-Outputs (Bilder, Logs), gitignored
  docs/          Projektplan, Datenzugriffs-Notizen
  requirements.txt
  PATCHES.md     Änderungen am vendorten Simulator-Code, mit Begründung
```

## Design-Entscheidungen

**Simulator: gym-pybullet-drones, Tag `v1.0.0`.** Die aktuelle
Hauptversion braucht Python >=3.12; `v1.0.0` läuft mit den zwei Patches
unten auch noch auf 3.12 (verifiziert) und bietet mit `VisionAviary` genau
das Nötige: direkte RPM-Ansteuerung aller
vier Rotoren plus Kamerabild aus Drohnensicht, ohne eingebauten Flugregler
im Weg. Details und die zwei nötigen Kompatibilitäts-Patches (Python
3.10/aktuelles NumPy):
[`PATCHES.md`](PATCHES.md). Alternativen (Webots, ArduPilot+Gazebo, Isaac
Sim, eigener Simulator) und warum sie verworfen wurden:
[`docs/projektplan.md`](docs/projektplan.md).

**Kein eigener Konnektom-Sim-Code aus boat.horse/fly übernommen.**
boat.horse/fly veröffentlicht keinen eigenen Quellcode für die dort
gezeigte Pipeline. `connectome/` ist komplett eigenständig geschrieben.
Verwandte, aber unabhängige Referenzimplementierungen (DOOMFLY,
eonsystems/fly-brain u. a.) sind im Projektplan verlinkt.

## Datenquellen & Lizenzen

- **MaleCNS-Konnektomdaten**: Janelia FlyEM + Google Research,
  [male-cns.janelia.org](https://male-cns.janelia.org/), Lizenz **CC-BY**
  (Attribution erforderlich).
- **gym-pybullet-drones**: [utiasDSL/gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones),
  MIT-Lizenz, vendort unter `simulator/` (siehe [`PATCHES.md`](PATCHES.md)).
- Eigener Code in diesem Repo: bisher ohne explizite Lizenz.

## Weiterführende Dokumentation

- [`docs/projektplan.md`](docs/projektplan.md) -- vollständiger Projektplan, Architektur, Simulator-Evaluation
- [`docs/connectome-data-access.md`](docs/connectome-data-access.md) -- Datenzugriff, Zelltyp-Recherche, Kalibrierungs-Historie
- [`PATCHES.md`](PATCHES.md) -- Änderungen am vendorten Simulator-Code
