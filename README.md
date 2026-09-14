# fly-drone-sim

Fliegen-Konnektom-Simulation (Fly Rig Anatomy Projekt, boat.horse/fly) steuert
eine simulierte Drohne statt eines Strandbeest-Laufroboters. Vollstaendig in
Simulation, kein Echtflug geplant.

Voller Plan: [`docs/projektplan.md`](docs/projektplan.md).

## Stand

- [x] M0 -- Simulator gewaehlt und Sanity-Check bestanden: `scripts/sanity_hover.py`
      steuert 4 Rotoren per direktem RPM-Wert an und liest ein RGB-Kamerabild
      aus der Drohnen-Perspektive aus (`files/sanity/frame_0.png`).
- [ ] M1 -- Kamerabilder offline durch den Medulla-Encoder + Konnektom-Sim
      schicken (noch kein geschlossener Kreis).
- [ ] M2 -- Geschlossener Kreis, Phase A (2 Freiheitsgrade: Schub + Gier aus
      Motor-Neuron-Pool-Summe/-Differenz, Roll/Pitch klassisch stabilisiert).
- [ ] M3 -- Tuning (Gain-Kalibrierung Spike-Rate <-> RPM-Offset).
- [ ] M4 -- Phase B, 4 Freiheitsgrade.
- [ ] M5 -- Auswertung (Stretch): emergentes Looming-Ausweichen / Hoehenhaltung?

## Ordnerstruktur

```
flydrone/
  simulator/     vendorter gym-pybullet-drones v1.0.0 (siehe PATCHES.md)
  scripts/       Sanity-/Test-Skripte, spaeter Trainings-/Eval-Skripte
  connectome/    noch leer -- hier entsteht die Medulla-Encoder- und
                 Konnektom-Anbindung (Neuronen-Sim laeuft auf dem
                 GPU-Server, nicht auf diesem Rechner)
  files/         Simulator-Outputs (Bilder, Logs) -- keine Quelldateien
  docs/          Projektplan und weitere Notizen
  requirements.txt
  PATCHES.md     Aenderungen am vendorten Simulator-Code, mit Begruendung
```

## Setup (dieser Rechner)

```
python3 -m venv venv
source venv/bin/activate   # oder venv\Scripts\activate unter Windows
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
