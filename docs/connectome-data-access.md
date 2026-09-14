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
