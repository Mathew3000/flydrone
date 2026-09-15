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
| M3 | Feintuning (Gain-Kalibrierung Spike-Rate <-> RPM-Offset) | ⏳ offen |
| M4 | Phase B, 4 Freiheitsgrade | ⏳ offen |
| M5 | Auswertung (Stretch): emergentes Looming-Ausweichen / Höhenhaltung | ⏳ offen |

**Aktuelles Ergebnis (M2, `scripts/m2_real_connectome.py`):** Die Drohne
hält mit dem echten MaleCNS-Konnektom (T4/T5 -> HS/H1/H2, 13.597 Neuronen,
207.059 Synapsen -- nicht dem synthetischen Testnetz) stabil die Höhe und
reagiert im Gier-Kanal sichtbar auf das, was die Kamera an Bewegung sieht.
Details, inklusive der Gewichts-Kalibrierung, die dafür nötig war:
[`docs/connectome-data-access.md`](docs/connectome-data-access.md).

## Voraussetzungen

- Python 3.10 (siehe [Design-Entscheidungen](#design-entscheidungen))
- ~30 MB Platz für den vendorten Simulator, optional ~1.1 GB für die echten
  Konnektom-Rohdaten

## Installation

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
bash scripts/setup_simulator.sh   # klont + patcht simulator/ (gitignored, ~28MB)
pip install -r requirements.txt
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
| `scripts/m1_offline_test.py` | M1: Kamera-Sequenz offline durch die Pipeline schicken | synthetisch |
| `scripts/m2_closed_loop.py` | M2: geschlossener Regelkreis | synthetisch |
| `scripts/m2_real_connectome.py` | M2: geschlossener Regelkreis | **echtes MaleCNS-Konnektom** (braucht die Rohdaten, siehe unten) |

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
Hauptversion braucht Python >=3.12; `v1.0.0` läuft mit Python 3.10 und
bietet mit `VisionAviary` genau das Nötige: direkte RPM-Ansteuerung aller
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
