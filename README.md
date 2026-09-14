# fly-drone-sim

Fliegen-Konnektom-Simulation (Fly Rig Anatomy Projekt, boat.horse/fly) steuert
eine simulierte Drohne statt eines Strandbeest-Laufroboters. Vollstaendig in
Simulation, kein Echtflug geplant.

Voller Plan: [`docs/projektplan.md`](docs/projektplan.md).

## Stand

- [x] M0 -- Simulator gewaehlt und Sanity-Check bestanden: `scripts/sanity_hover.py`
      steuert 4 Rotoren per direktem RPM-Wert an und liest ein RGB-Kamerabild
      aus der Drohnen-Perspektive aus (`files/sanity/frame_0.png`).
- [x] M1 -- Kamerabilder offline durch Medulla-Encoder (`connectome/medulla_encoder.py`,
      optischer Fluss via OpenCV) + eine LIF-Konnektom-Simulation
      (`connectome/lif_network.py`, scipy.sparse) geschickt und Motor-Neuron-
      Output geloggt (`scripts/m1_offline_test.py`, `connectome/tests/test_toy_network.py`
      gegen ein synthetisches Testnetz). Zusaetzlich jetzt mit dem echten
      MaleCNS-Konnektom validiert: `connectome/tests/test_real_connectome.py`
      (siehe M2-Eintrag unten und "Echte Konnektom-Daten" weiter unten).
- [x] M2 -- Geschlossener Kreis, Phase A:
      - `scripts/m2_closed_loop.py` -- synthetisches Testnetz. Kamera ->
        Medulla-Encoder -> LIF-Sim -> Motor-Decoder steuert Gier- und
        Hoehen-Sollwert; Roll/Pitch/Position haelt gym-pybullet-drones'
        eigener DSLPIDControl-Regler (bewusst wiederverwendet statt selbst
        gebaut).
      - `scripts/m2_real_connectome.py` -- **echtes MaleCNS-Konnektom**
        (T4/T5 -> HS/H1/H2, der reale optomotorische Kurskorrektur-Pfad,
        aus den lokalen Bulk-Daten, siehe unten). Ergebnis (14.09.2026):
        Drohne steigt stabil auf Zielhoehe (roll/pitch bleiben ~0, z
        pendelt sich bei 1.005 m ein) und der Gier-Sollwert driftet
        sichtbar (~2.7°) mit dem, was die Kamera an Bewegung sieht, waehrend
        des Steigflugs -- sobald sie ruhig schwebt, faellt auch der
        Netzwerk-Output wieder auf ~0. **Damit haelt die echte
        Fliegen-Konnektom-Simulation die Drohne tatsaechlich in der Luft.**
        Musste dafuer extra kalibriert werden -- die rohen
        Synapsenanzahl-Gewichte aus MaleCNS sind ~1000x staerker als das
        Testnetz und saettigen ungefiltert beide Motor-Pools gleichzeitig
        auf den Maximalwert (siehe `WEIGHT_SCALE` in
        `connectome/tests/test_real_connectome.py`'s Docstring fuer Details).
- [ ] M3 -- Tuning (Gain-Kalibrierung Spike-Rate <-> RPM-Offset).
- [ ] M4 -- Phase B, 4 Freiheitsgrade.
- [ ] M5 -- Auswertung (Stretch): emergentes Looming-Ausweichen / Hoehenhaltung?

## Ordnerstruktur

```
flydrone/
  simulator/     vendorter gym-pybullet-drones v1.0.0 (siehe PATCHES.md)
  scripts/       Sanity-/Test-Skripte, spaeter Trainings-/Eval-Skripte
  connectome/    Medulla-Encoder, LIF-Netzwerk-Engine, Motor-Decoder, Daten-Loader:
                 - data_loader.py: synthetisches Testnetz + neuPrint-API-Anbindung
                 - local_maleCNS.py: laedt echte MaleCNS-Konnektom-Daten direkt
                   aus den lokalen Bulk-Dateien (kein Account/Token noetig,
                   keine Netzwerk-Query-Limits) -- siehe "Echte Konnektom-Daten"
                   weiter unten
  data/raw/      Lokale MaleCNS-Bulk-Dateien (gitignored, siehe unten) --
                 nicht im Repo, muessen einmalig heruntergeladen werden
  files/         Simulator-Outputs (Bilder, Logs) -- keine Quelldateien
  docs/          Projektplan und weitere Notizen
  requirements.txt
  PATCHES.md     Aenderungen am vendorten Simulator-Code, mit Begruendung
```

## Echte Konnektom-Daten: lokale Bulk-Dateien (kein Account noetig)

Die MaleCNS-Rohdaten (Janelia FlyEM + Google Research, CC-BY-lizenziert)
gibt es als direkten, oeffentlichen Download -- kein neuPrint-Account/Token
noetig (das ist nur fuer einzelne API-Queries erforderlich, siehe
`docs/connectome-data-access.md`). Zwei Dateien werden gebraucht: die
Body-Annotationen (~13 MB, Zelltyp/Seite pro Neuron) und die komplette
Konnektivitaets-Gewichtsmatrix (~1.1 GB, ~152 Mio. Zeilen).

**Download (PowerShell):**

```powershell
cd flydrone
mkdir data\raw -Force
$ProgressPreference = 'SilentlyContinue'  # sonst bremst Invoke-WebRequest/curl-Alias massiv
curl.exe -L -o data\raw\body-annotations-male-cns-v1.0-minconf-0.5.feather `
  https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather
curl.exe -L -o data\raw\connectome-weights-male-cns-v1.0-minconf-0.5.feather `
  https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather
```

**Download (Linux/macOS):**

```bash
mkdir -p data/raw
curl -L -o data/raw/body-annotations-male-cns-v1.0-minconf-0.5.feather \
  https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather
curl -L -o data/raw/connectome-weights-male-cns-v1.0-minconf-0.5.feather \
  https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather
```

`curl.exe` explizit (nicht `curl`, das in PowerShell ein Alias fuer
`Invoke-WebRequest` ist und bei grossen Dateien wegen der Fortschrittsbalken-
Anzeige stark bremst).

Danach direkt nutzbar ohne Netzwerkzugriff, z. B.:

```bash
pip install pyarrow pandas scipy   # falls noch nicht in requirements.txt-Umgebung installiert
python -c "from connectome.local_maleCNS import local_fetch_lr_subnetwork; n = local_fetch_lr_subnetwork(); print(n['n_neurons'], 'Neuronen,', n['weights'].nnz, 'Synapsen')"
python scripts/m2_real_connectome.py   # echtes Konnektom haelt die Drohne in der Luft
```

Alle offiziellen Download-Links: https://male-cns.janelia.org/download/

`connectome/local_maleCNS.py` liest die ~1.1 GB grosse Gewichtsdatei
speicherschonend im Batch-Streaming-Verfahren (nie das ganze File auf einmal
in den Speicher) -- Details im Docstring dort.

## Echte Konnektom-Daten (neuPrint-API) einrichten

Fuer einzelne Ad-hoc-Queries (z. B. neue Zelltypen erkunden) gibt es
zusaetzlich die neuPrint-API. `.env.example` nach `.env` kopieren (im
Projekt-Wurzelverzeichnis) und den neuPrint-API-Token eintragen -- `.env`
ist gitignored, landet also nie im Repo. `connectome/data_loader.py` liest
sie automatisch ein, kein manuelles Setzen von Windows-Umgebungsvariablen
noetig. Details: `docs/connectome-data-access.md`.

## Setup (dieser Rechner)

```
python3 -m venv venv
source venv/bin/activate   # oder venv\Scripts\activate unter Windows
bash scripts/setup_simulator.sh   # klont + patcht simulator/ (gitignored, ~28MB)
pip install -r requirements.txt
PYTHONPATH=simulator python scripts/sanity_hover.py
```

Erwartete Ausgabe: `HOVER_RPM`/`MAX_RPM`-Werte, eine Positionsangabe nach 3s
simulierter Zeit, und `files/sanity/frame_0.png` (ein 48x64-Kamerabild aus
Drohnensicht).

## Warum v1.0.0 statt der aktuellen Simulator-Version?

Die derzeitige Hauptversion von gym-pybullet-drones braucht Python >=3.12.
v1.0.0 (die Version aus dem urspruenglichen Paper) laeuft mit Python 3.10 und
bietet mit `VisionAviary` genau das, was wir brauchen: direkte RPM-Ansteuerung
aller vier Rotoren plus Kamerabild aus Drohnensicht, ohne eingebauten
Flugregler im Weg. Details und die zwei noetigen Kompatibilitaets-Patches in
[`PATCHES.md`](PATCHES.md). Ein spaeteres Upgrade auf die aktuelle Version
(z. B. sobald auf Python 3.12 gewechselt wird) ist moeglich und wuerde die
Patches ueberfluessig machen.

## Wichtig: kein eigener Neuronen-Sim-Code hier (noch)

boat.horse/fly veroeffentlicht keinen eigenen Quellcode fuer die dort
beschriebene Konnektom-Pipeline. Die im Projektplan erwaehnten Repos
(DOOMFLY, eonsystems/fly-brain, u.a.) sind verwandte, aber eigenstaendige
Implementierungen -- kein Fork-Ziel, sondern Referenzmaterial fuer den
Medulla-Encoder/Motor-Decoder-Teil, der noch geschrieben werden muss.
