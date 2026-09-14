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
