# Nächste Schritte

Übergabedokument. Der Stand steht in der Statustabelle der [README](../README.md),
die Arbeitsregeln in [CLAUDE.md](../CLAUDE.md). Hier steht, **was als nächstes zu
tun ist und warum** — mit den Messwerten, die die Reihenfolge begründen, damit
niemand sie noch einmal erheben muss.

Kurzfassung des Stands: Die Pipeline Kamera → Encoder → LIF-Netz (echtes
MaleCNS-Konnektom) → Decoder → Flugregler ist vollständig und läuft. Nachgewiesen
sind die Optomotorik-Reaktion (richtungsselektiv, umkehrend), der Looming-Transient
mit Winkelgrößen-Schwelle, das Auslesen von Roll aus dem vertikalen System, und
freier Flug mit Zentrierreaktion im Labyrinth. 19 Tests grün.

---

## Priorität 1 — Retinotope Rezeptivfelder

**Das ist der einzige Punkt, an dem mehrere unabhängige Sackgassen zusammenlaufen.**
Alle Encoder in `connectome/medulla_encoder.py` mitteln über **Bildhälften** und
geben jedem Neuron einer Population denselben Wert. LC4/LPLC2 und T4/T5 sind in
Wirklichkeit kolumnäre Zellen mit **kleinen Rezeptivfeldern**; die Population
kachelt das Blickfeld.

### Was das kostet, gemessen

**(a) Eine Wand voraus ist unsichtbar.** Bei Vorwärtsflug ist das Flussfeld eine
radiale Expansion um die Flugrichtung — die Wand, auf die man zufliegt, liegt im
**Expansionsfokus, wo die Bildbewegung definitionsgemäß null ist**. Gleichzeitig
dominieren die Seitenwände bei konstantem Abstand jede globale Mittelung. Drive
gegen Wandabstand im Korridorflug (`scripts/m6_maze.py`):

| Auslesung | 8–12 m | 0.6–1 m |
|---|---|---|
| global radial (Looming) | 0.0014 | 0.0039, Maximum bei 10.8 m |
| nur frontales Drittel | 0.0001 | **exakt 0** ab 1.5 m |
| seitliches Gleichgewicht | 0.0036 | 0.0104, monoton |

**(b) Ein lokalisiertes Objekt wird weggemittelt.** Ein anfliegendes Objekt füllt
6–35° eines 60°-Blickfelds. Sein Signal konkurriert im Hälftenmittel mit dem
statischen Hintergrund, während ein Vollfeld-Reiz (Drehung) das ganze Bild bewegt.

### Was zu bauen ist

Ein Raster (Vorschlag: 4×8 Patches auf dem 48×64-Bild), jedes Patch ein
Rezeptivfeld. Die ~300 LC-Zellen bzw. ~13.500 T4/T5-Zellen werden auf Patches
verteilt, statt alle denselben Wert zu bekommen. Die Zuordnung sollte aus den
Annotationen kommen, nicht willkürlich sein — die Spalten `assignedOlHex1` und
`assignedOlHex2` in `body-annotations-*.feather` enthalten offenbar
Ommatidien-Koordinaten. **Das ist ungeprüft und der erste Schritt: nachsehen, ob
sich daraus eine echte retinotope Karte bauen lässt.** Falls ja, ist die Zuordnung
biologisch begründet statt erfunden.

### Akzeptanzkriterium

Der frontale Nahbereichskanal muss mit abnehmendem Wandabstand **monoton steigen**
statt innerhalb von 1.5 m auf 0 zu fallen. Danach — und erst danach — ist ein
erneuter Labyrinthtest aussagekräftig.

### Was es ausdrücklich NICHT löst

Ein 4×8-Raster wurde in M5 bereits gegen das Spezifitätsproblem des
Looming-Pfads getestet und half kaum (0.51–0.65x). Dessen Ursache war eine
andere — Rektifizierung pro Pixel vor dem Mitteln, siehe
`encode_to_drive_looming`. **Retinotopie ist kein Allheilmittel**, sie löst genau
das Lokalisierungsproblem oben.

---

## Priorität 2 — Labyrinth zu Ende fliegen

Aktuell: **Wegpunkt 6 von 18, 33 % der Strecke, 1 von 5 Kurven.** Die Drohne
bleibt bei Zelle (5,3) = Welt (−2.5, −2.5) stehen, wo sie nach Norden abbiegen
müsste; geradeaus ist Wand.

Hängt an Priorität 1. Ohne frontalen Kanal ist ein Wiederholungstest nur eine
Wiederholung desselben Ergebnisses. Erfolglos durchprobiert und **nicht noch
einmal nötig**:

- adaptive Bremse (Basislinien-Subtraktion) → 692–784 Kontaktbilder gegen 0
- `TURN_GAIN` 20 → 691 Kontakte, 45 → instabil bei 1040
- Gain-Sweep 60/150/400 → 400 ist das Optimum

Das langsame Kriechen ist **der Mechanismus, nicht der Fehler**: die Zentrierung
braucht Zeit, um die Drohne um eine Ecke zu schwenken.

---

## Priorität 3 — Kursstabilisierung gegen Störung

Billigster offener Punkt, alles Nötige existiert. Drohne fliegt geradeaus, eine
Seitenstörung dreht sie weg, der Gier-Kanal des optomotorischen Pfads
(`scripts/m3_optomotor.py`, `local_fetch_dn_subnetwork`) korrigiert. Messgröße:
Geradlinigkeit mit gegen ohne Konnektom. Das ist die eigentliche biologische
Funktion der Optomotorik-Reaktion.

## Priorität 4 — Anflug auf einen bewegten Reiz

Ein oszillierendes Objekt erzeugt Bewegung, der Gier-Kanal dreht darauf zu.
Machbar. **Ehrlich benennen:** ob Anziehung oder Abstoßung herauskommt, ist unsere
Vorzeichenwahl und nicht aus dem Konnektom ableitbar — wie bei jedem Kanal in
diesem Projekt.

Ein **statisches** Ziel bleibt für Bewegungsdetektoren unsichtbar. Das ist keine
Implementierungslücke, sondern T4/T5.

## Priorität 5 — Nicken in beide Richtungen

`encode_to_drive_vertical` rektifiziert, jede Population meldet nur
Abwärtsbewegung. Beide **Roll**-Richtungen sind abgedeckt (sie erregen
gegenüberliegende Seiten), bei **Nicken** erregt Nase-hoch beide und Nase-runter
keine. Bräuchte ein zweites Populationspaar für Aufwärtsbewegung; `somaSide` gibt
nur zwei Pools her, also müsste die Aufteilung woanders herkommen — vielleicht
ebenfalls retinotop (Priorität 1).

---

## Was in dieser Umgebung nicht funktioniert

**`python3` und `git` sind die Xcode-Varianten und verweigern den Dienst**, weil
die Lizenz nicht akzeptiert ist (Exit 69). Das ist mir eine ganze Weile nicht
aufgefallen, weil die Meldung in gefilterter Ausgabe untergeht — **zwei
Skript-Bearbeitungen liefen stillschweigend nicht, und ich habe danach zweimal
gemessen und den alten Zustand für ein Ergebnis gehalten.**

Ersatz bis jemand `sudo xcodebuild -license accept` ausführt:

- Python: `.venv/bin/python`
- Git: `/Library/Developer/CommandLineTools/usr/bin/git`

Wer ein Skript per Textersetzung bearbeitet: **immer mit `assert old in s`**, und
danach mit `grep` nachsehen, ob die Änderung wirklich drin ist. Genau diese
Prüfung hat den Fehler am Ende aufgedeckt.

## Prüfen, dass nichts kaputt ist

```bash
PYTHONPATH=. .venv/bin/python -m pytest connectome/tests/     # 19 passed
PYTHONPATH=simulator .venv/bin/python scripts/m3_optomotor.py # Umkehr, überlappungsfrei
PYTHONPATH=simulator .venv/bin/python scripts/m5_looming.py   # Kontrollen bei exakt 0
```

Die Tests laufen ohne Simulator (synthetische Reize) und decken die
Richtungsselektivität, die Looming-Spezifität, die Sättigungsgrenze und die
Existenz des VS-Typs ab. Sie sind die Stellen, an denen die teuer erkauften
Erkenntnisse festgeklemmt sind — wer sie weichklopft, verliert sie.
