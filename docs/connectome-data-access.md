# Woher die echten Konnektom-Daten kommen (noch nicht eingebunden)

boat.horse/fly veroeffentlicht keinen eigenen Quellcode fuer die dort
gezeigte Medulla-Encoder/Konnektom-Pipeline. `connectome/` in diesem Repo ist
komplett selbst geschrieben und laeuft bisher nur gegen ein synthetisches
Test-Netzwerk (`connectome.data_loader.make_toy_network()`). Dieser Abschnitt
haelt fest, wie die echten Daten reinkommen, sobald du sie hast.

## Quelle: MaleCNS via neuPrint

- Datensatz: https://male-cns.janelia.org/ (Janelia FlyEM + Google Research)
- Lizenz: **CC-BY** (Attribution erforderlich, sonst frei nutzbar)
- Zugriff: Account + API-Token auf neuprint.janelia.org anlegen (kostenlos,
  academic/community-Zugang), dann `pip install neuprint-python`
- Formate: Connectivity-Gewichte (~1.1 GB), Synapse-Partner-Daten (~6.8 GB),
  Body-Annotations/Zelltypen (~13 MB) -- als Apache-Arrow-Feather, CSV, oder
  komplette Neo4j-DB

## Setup

1. Account unter https://neuprint.janelia.org anlegen, API-Token kopieren
2. `export NEUPRINT_APPLICATION_CREDENTIALS=<dein-token>` (nicht ins Repo
   committen -- `.env`/Shell-Profil, nicht `.gitignore`-Ausnahme noetig, da
   ohnehin nicht im Repo)
3. `pip install neuprint-python`
4. `connectome/data_loader.py::fetch_from_neuprint(...)` ausfuehren -- Skizze
   ist im Code vorhanden, aber ungetestet (kein Token verfuegbar in dieser
   Session)

## Welche Zelltypen zuerst?

Aus der Literatur gut belegt (siehe Kommentare in `connectome/medulla_encoder.py`):

- **T4 / T5** -- die elementaren Bewegungsdetektoren der Fliege. T4 reagiert
  auf Helligkeitszunahmen, T5 auf -abnahmen; je vier Subtypen (a-d) fuer die
  vier Grundrichtungen. Quelle u. a. die eLife-Arbeit zum "ON motion
  detection"-Schaltkreis.
- **LC4 / LPLC2** -- Looming-Detektoren (Kollisionswarnung). LPLC2 gilt laut
  Klapoetke et al. 2017 als "ultra-selektiv" fuer sich naehernde Objekte
  ueber radiale Bewegungs-Opponenz.

**Offen:** ob diese Zelltypen im MaleCNS-Datensatz exakt so benannt sind wie
im (aelteren) Hemibrain-Datensatz (z. B. "T4a" vs. eine andere Konvention).
Muss beim ersten echten `fetch_neurons(...)`-Aufruf verifiziert werden --
`fetch_from_neuprint()` nimmt deshalb eine Liste von Typ-Strings/Regex
entgegen statt sie hart zu kodieren.

## Warum das noch nicht laeuft

Kein neuPrint-Token in dieser Session verfuegbar (und sollte auch nicht ueber
den Chat geteilt werden -- lieber lokal als Umgebungsvariable setzen). Sobald
du einen Account hast, ist der naechste Schritt: `fetch_from_neuprint(["T4.*",
"T5.*", "LC4", "LPLC2"])` probeweise laufen lassen, pruefen ob die
Zelltyp-Strings matchen, und die zurueckgegebene `weights`-Matrix anstelle von
`make_toy_network()` in `scripts/m1_offline_test.py` einsetzen.

## Bekannte Einschraenkung: neuPrint von hier aus nicht erreichbar

Getestet mit echtem Token (14.09.2026): `pip install neuprint-python` klappt
(PyPI ist erreichbar), aber jeder Request an `neuprint.janelia.org` scheitert
mit `ProxyError ... 403 Forbidden` -- die Netzwerk-Freigabe der Cowork-
Sandbox-Umgebung, in der Claude hier arbeitet, laesst nur eine bestimmte
Domain-Allowlist durch (GitHub/PyPI funktionieren, neuPrint nicht).

`fetch_from_neuprint()` in `connectome/data_loader.py` ist fertig und sollte
funktionieren, muss aber von einem Rechner mit normalem Internetzugang
ausgefuehrt werden -- z. B. direkt auf deinem Windows-Rechner (ausserhalb
dieser Sandbox) oder auf dem GPU-Server, wo ohnehin die eigentliche
166k-Neuronen-Simulation laufen soll. Kurztest zum Ausprobieren:

```
cd flydrone
pip install -r requirements.txt
python -c "from connectome.data_loader import fetch_from_neuprint; d = fetch_from_neuprint(['T4.*','T5.*','LC4','LPLC2']); print(d['n_neurons'], list(d['cell_type_indices'].keys()))"
```

Falls das laeuft: Ergebnis (Anzahl Neuronen, welche Zelltyp-Keys nicht leer
sind) zurueckmelden, dann passe ich die Cell-Type-Regex in
`medulla_encoder.py`/`data_loader.py` bei Bedarf an die tatsaechliche
MaleCNS-Namenskonvention an.

## Zwischenstand: echte Daten abgerufen (14.09.2026)

`fetch_from_neuprint(['T4.*','T5.*','LC4','LPLC2'])` funktioniert (Fix siehe
Commit-History): 13.896 Neuronen total (T4: 6865, T5: 6720, LC4: 126,
LPLC2: 185), Datensatz `male-cns:v1.0`.

`explore_downstream_types(['LC4','LPLC2'])` fand als staerkste echte Ziele
u. a. **DNp01** (Giant Fiber, 2 Zellen -- exakt wie in der Literatur), sowie
DNp02/04/06/11/103, DNg40 -- das ist der bekannte visuelle
Fluchtreflex-Schaltkreis (Sprung + Fluginitiierung).

`explore_downstream_types(['DNp01','DNp02','DNp04','DNp06','DNp11','DNp103','DNg40'])`
zeigte danach nur noch schwach gewichtete AN-/IN-Zelltypen (aufsteigend bzw.
lokale Zwischenneuronen) -- vermutlich weil die eigentliche
Giant-Fiber-Ausgangssynapse (auf das Sprungmuskel-Motoneuron TTMn) in der
Literatur als teilweise **elektrisch** beschrieben ist und in rein
chemisch-synapsenbasierten Gewichten hier untergewichtet erscheint.

**Kurswechsel:** Der Fluchtreflex (LC4/LPLC2 -> Giant Fiber) ist ein
ballistischer Ein/Aus-Trigger, kein abgestuftes Signal -- fuer die
kontinuierliche Gier-/Kurssteuerung, die wir fuer die Drohne brauchen, ist
der T4/T5 -> optomotorischer Kurskorrektur-Pfad (Horizontal-/Vertical-System-
Zellen etc.) biologisch die passendere Wahl. Naechster Schritt: einzelne
T4/T5-Subtypen (z. B. T4a/T5a) statt aller vier Subtypen auf einmal, um die
Downstream-Abfrage handhabbar zu halten.

## Alternative: lokale Bulk-Dateien statt neuPrint-API (14.09.2026)

Statt einzelner neuPrint-Queries (die von dieser Sandbox aus ohnehin nicht
erreichbar sind, siehe oben) gibt es die kompletten MaleCNS-Rohdaten auch als
direkten, oeffentlichen Download -- kein Account/Token noetig, keine
Query-Groessenlimits. Download-Befehle: siehe README.md, Abschnitt "Echte
Konnektom-Daten: lokale Bulk-Dateien".

Zwei Dateien, beide Apache-Arrow-Feather:
- `body-annotations-male-cns-v1.0-minconf-0.5.feather` (~13 MB): eine Zeile
  pro Neuron -- `bodyId`, `type`, `instance`, `somaSide` (L/R/M/None; die
  reale anatomische Links-Rechts-Zuordnung), u. a.
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (~1.1 GB, ~152 Mio.
  Zeilen): `body_pre`, `body_post`, `weight` -- das komplette
  CNS-Konnektivitaets-Netz, nicht auf einen Zelltyp gefiltert.

**Speicher-Hinweis:** `pd.read_feather()` auf die 1.1-GB-Datei OOM-killt den
Prozess auf einer 3.8-GB-RAM-Maschine ohne Swap (bestaetigt). Loesung:
`connectome/local_maleCNS.py` liest sie stattdessen ueber PyArrows
IPC-Batch-Reader (`pa.ipc.open_file(...).get_batch(i)`), Batch fuer Batch
(~65536 Zeilen), verworfen nach dem Filtern -- scannt alle 152 Mio. Zeilen in
~5s bei flachem Speicherverbrauch. Nicht durch ein volles `read_feather()`
ersetzen, ohne den Speicherverbrauch neu zu pruefen.

`connectome/local_maleCNS.py` bietet:
- `local_downstream_types(...)`: netzwerkfreies Aequivalent zu
  `explore_downstream_types()`, ohne Query-Groessenlimit
- `local_fetch_subnetwork(cell_types)`: Subnetz (nur interne Konnektivitaet
  zwischen den angegebenen Typen), gleiche Rueckgabeform wie
  `fetch_from_neuprint()`
- `local_fetch_lr_subnetwork(...)`: wie oben, zusaetzlich nach `somaSide`
  (L/R) in `visual_L`/`visual_R`/`motor_left`/`motor_right`-Populationen
  aufgeteilt -- direkter Drop-in-Ersatz fuer `make_toy_network()`

## Der echte Kurskorrektur-Pfad: T4/T5 -> HS/H1/H2 (bestaetigt lokal, 14.09.2026)

`local_downstream_types(["T4.*", "T5.*"])` gegen die kompletten lokalen Daten
(keine Query-Limits) bestaetigte: die Horizontal-System-Zellen **HSE, HSN,
HSS, HST** sowie **H1, H2** (Lobula-Plate-Tangentialzellen) erhalten **>90%**
ihres modellierten Eingangs von T4/T5 -- das ist der reale, in der Literatur
gut belegte optomotorische Kurskorrektur-Pfad der Fliege (im Gegensatz zum
LC4/LPLC2 -> Giant-Fiber-Fluchtreflex, siehe oben -- ballistischer
Ein/Aus-Trigger, ungeeignet fuer kontinuierliche Gier-Steuerung).

Damit ergibt `local_fetch_lr_subnetwork()` (Default-Parameter) ein echtes
Subnetz: **13.597 Neuronen, 207.059 Synapsen**
(`visual_L`: 6790, `visual_R`: 6795, `motor_left`: 6, `motor_right`: 6 --
HS/H1/H2 hat insgesamt nur 12 Zellen, aber jede davon ist eine reale,
gut charakterisierte Tangentialzelle mit breitem rezeptivem Feld, kein
Kompromiss).

## Kalibrierungs-Bug: reale Gewichte saettigen beide Motor-Pools (gefunden + behoben, 14.09.2026)

Erster Test von `encode_to_drive_hemifield()` + `LIFNetwork` mit dem echten
Subnetz (bei `gain=8.0`, dem fuer das Testnetz kalibrierten Wert) ergab fuer
**jeden** Stimulus (auch reine Links-Bewegung) exakt `motor_left=0.2500,
motor_right=0.2500` -- der `master=0.25`-Deckel aus
`motor_decoder.decode_pool()`, also volle Saettigung auf beiden Seiten
unabhaengig vom Reiz.

Ursache (gefunden per Gain/Gewichts-Sweep, siehe
`connectome/tests/test_real_connectome.py`): die realen MaleCNS-Gewichte sind
rohe Synapsenanzahlen (Mittelwert ~2.6, Maximum 56), nicht die kleinen
Zufallsgewichte (~0.15) des Testnetzes. Jede der 6 Motor-Zellen pro Seite
erhaelt in Summe **>12.000** Gewichtseinheiten von den ~6.800
Sehsystem-Neuronen der jeweiligen Seite -- selbst schwache Aktivitaet dort
reicht, um die Motor-Zellen sofort auf Maximalrate zu treiben, unabhaengig
von der tatsaechlichen Staerke/Seite des Reizes.

**Fix:** `WEIGHT_SCALE = 0.0015` -- reale Gewichte vor dem Aufbau des
`LIFNetwork` mit diesem Faktor multiplizieren (das reale-Daten-Aequivalent zu
den kleinen Testnetz-Gewichten). Nach einem Sweep ueber Gewichts-Skalierung x
Encoder-Gain bestaetigt: bei `WEIGHT_SCALE=0.0015, GAIN=8.0` liefert reine
Links-Bewegung `motor_left≈0.25, motor_right≈0.006` (starke, saubere
Asymmetrie), reine Rechts-Bewegung das Spiegelbild, und symmetrische Bewegung
liefert `motor_left≈motor_right` (kein systematischer Links/Rechts-Bias durch
die leicht unterschiedliche Neuronenzahl pro Seite). Alle drei Faelle sind
jetzt Regressionstests in `connectome/tests/test_real_connectome.py`.

## M2 mit echtem Konnektom: Drohne bleibt in der Luft (14.09.2026)

`scripts/m2_real_connectome.py` (Kopie von `m2_closed_loop.py`, aber mit
`local_fetch_lr_subnetwork()` + `encode_to_drive_hemifield()` +
`WEIGHT_SCALE=0.0015` statt des Testnetzes) laeuft stabil: Drohne steigt in
~1s auf Zielhoehe (1.0 m), roll/pitch bleiben bei 0.000 rad, z pendelt sich
bei 1.005 m ein. Der Gier-Sollwert driftet waehrend des Steigflugs (durch die
kamera-sichtbare Eigenbewegung) auf ~0.047 rad (~2.7°) und bleibt danach
konstant, sobald keine Bildbewegung mehr da ist -- genau das erwartete
Verhalten, jetzt mit dem echten Fliegen-Konnektom statt einem synthetischen
Platzhalter.
