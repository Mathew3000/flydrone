# Projektplan: Fliegen-Konnektom steuert eine Drohne (Simulation)

Stand: September 2026
Basis: [boat.horse/fly/](https://boat.horse/fly/) — Fly Rig Anatomy Project (166.700 LIF-Neuronen, MaleCNS-Konnektom, aktuell an Strandbeest + LED-Panel)

## 1. Zielsetzung

Das bestehende Konnektom-Sim-Pipeline (Kamera → Medulla-Encoder → 166.700-Neuronen-Simulation → 815 Motor-Neuronen → Decoder → Aktuator) soll statt eines Strandbeest-Laufroboters eine simulierte Drohne steuern. Ziel der ersten Phase ist **kein** Flugverhalten in echter Hardware, sondern ein geschlossener Regelkreis vollständig in Software: virtuelle Kamera liefert Bild, Konnektom-Sim verarbeitet es, Motor-Ausgabe treibt eine simulierte Quadrocopter-Physik, deren simulierte IMU wieder zurück in die Balance-/Propriozeptions-Pfade des Netzwerks fließt.

Bewusst ausgeklammert: eigene Physik-/Renderingengine bauen. Das übernimmt ein fertiger Simulator (Entscheidung siehe Abschnitt 2).

## 2. Simulator-Entscheidung

**Empfehlung: [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)** (MIT-Lizenz, Python, aktiv gepflegt — u. a. 2026 im GitHub Maintainer Spotlight, Gymnasium-kompatibel).

Geprüfte Eckdaten (direkt aus dem Quellcode verifiziert, nicht nur aus der Doku):

- `ActionType.RPM` — direkte Ansteuerung jedes einzelnen Rotors per RPM-Wert. Kein Umweg über einen eingebauten Flugregler nötig; die Motor-Decoder-Ausgabe kann 1:1 auf die vier Rotor-RPMs geschrieben werden, genau wie sie aktuell die Strandbeest-Servos per PWM ansteuert.
- `ObservationType.RGB` / `DEP` — First-Person-Kamerabild (RGB bzw. Tiefe) aus Sicht jeder simulierten Drohne. Das ersetzt die Webcam 1:1 als Eingang für den bestehenden Medulla-Encoder (optischer Fluss).
- Läuft headless über PyBullets DIRECT-Modus — kein Display nötig, damit problemlos auf deinem Server statt am Desktop.
- Python 3.10, reine pip/conda-Installation, keine Lizenzserver oder GUI-Abhängigkeiten wie bei Gazebo/ROS.
- Multi-Agent-fähig, falls später mehrere Drohnen oder Parametervarianten parallel getestet werden sollen.

**Verworfene Alternativen:**

- *Webots* (Mavic-2-Pro-Modell vorhanden, Python-API) — mächtiger und "batteries included", aber deutlich schwerer (eigene Laufzeitumgebung, IDE-zentriert), und die Kamera-/Motorschnittstelle ist stärker auf ein bestehendes Flugregler-Framework zugeschnitten. Für "rohe RPM rein, Bild raus" ist gym-pybullet-drones direkter.
- *ArduPilot SITL + Gazebo* — am realistischsten (echter Autopilot-Stack, MAVLink), aber damit holst du dir ein ganzes zweites Projekt (SITL-Toolchain, Gazebo-Weltenbau, MAVLink-Bridging) ins Haus — genau das, was du ausschließen wolltest.
- *NVIDIA Isaac Sim* — overpowered für die Fragestellung, GPU-/Treiber-Anforderungen (RTX, Omniverse) passen nicht zu deinen M40-Karten (keine RT-Cores, Maxwell-Generation).
- Eigener minimalistischer Quadrotor-Punktmasse-Simulator in Python — bewusst verworfen, das wäre exakt das "Simulator als eigenes Projekt", das du vermeiden willst.

Fallback, falls gym-pybullet-drones sich in der Praxis als zu limitiert erweist (z. B. Kameraauflösung/-latenz nicht ausreichend für den optischen Fluss): Webots als zweite Wahl, da dort ebenfalls eine fertige Drohne samt Kamera existiert und nur die Anbindung getauscht werden müsste.

## 3. Architektur

```
[PyBullet-Physik: Quadrocopter]
      │ RGB-Kamera (Drohnen-POV)         │ simulierte IMU (Gyro/Accel)
      ▼                                  ▼
[Medulla-Encoder]                  [Propriozeptions-/Balance-Eingang]
      │ optischer Fluss → Bewegungs-/Looming-Detektoren
      ▼
[Konnektom-Simulation: 166.700 LIF-Neuronen, GeNN/Numba]
      │ 815 Motor-Neuronen (Spikes/20 ms)
      ▼
[Motor-Decoder: Pool-Auswahl → Mixer → 4× Rotor-RPM]
      ▼
[PyBullet: ActionType.RPM] ──── schließt den Kreis
```

Das ist strukturell exakt die bestehende Pipeline — nur werden Strandbeest-Servos durch `env.step(rpm_array)` und die Webcam durch `env._getDroneImages()` (bzw. das Äquivalent in der jeweils installierten Version) ersetzt.

## 4. Sensor-Mapping (Kamera → Medulla-Encoder)

Unverändert übernehmbar: Der Medulla-Encoder erwartet ein Kamerabild und erzeugt daraus optischen Fluss für Bewegungs-/Looming-Detektoren, nicht für die Retina direkt (das war ja der entscheidende Fix im Originalprojekt). Die RGB-Observation aus gym-pybullet-drones liefert genau das Rohbild, das an derselben Stelle in die Pipeline eingespeist wird wie vorher das Webcam-Frame.

Zwei Dinge vorab klären:
- **Auflösung/Framerate:** PyBullets Kamera-Rendering ist deutlich langsamer als eine echte Webcam auszulesen. Realistisch niedrige Auflösung (z. B. 64×64 oder 84×84, wie in RL-Setups mit dieser Simulator-Familie üblich) wählen und prüfen, ob der Medulla-Encoder damit noch sinnvolle Flussfelder erzeugt.
- **Blickrichtung:** Für "wie eine Fliege" macht eine nach vorne/unten geneigte Kamera (ventraler optischer Fluss für Höhenhaltung, frontaler für Looming/Hindernisse) mehr Sinn als eine reine Nadir-Kamera.

## 5. Motor-Mapping (Konnektom → Rotoren)

Das ist der Kernpunkt, an dem "biologisch sauber" und "schnell zum Laufen bringen" auseinanderlaufen. Empfehlung: in Stufen vorgehen, nicht gleich das Maximum versuchen.

**Phase A – Pragmatisch, 2 Freiheitsgrade (empfohlener Start):**
Genau wie beim Strandbeest zwei Pool-Mittelwerte bilden (z. B. linke/rechte Flügel-/Halteren-Motor-Neuronen, 88 Neuronen insgesamt, oder testweise erstmal die bereits decodierten Bein-Pools weiterverwenden) und daraus zwei Kanäle ableiten: **Summe** → Kollektivschub/Höhe, **Differenz** → Gier (Yaw). Roll/Pitch werden vorerst von einem einfachen, klassischen PD-Regler stabilisiert (kein Konnektom-Anteil). Das ist bewusst nicht "die Fliege fliegt die Drohne komplett", sondern ein erster geschlossener Kreis, der zeigt, dass Netzwerk-Output überhaupt sinnvoll auf Flugverhalten wirkt.

**Phase B – Erweiterung auf 4 Freiheitsgrade:**
Sobald Phase A stabil läuft, zusätzliche Pools (z. B. vordere/hintere Anteile der Flügel-Motor-Neuronen, falls im Konnektom so unterscheidbar) für Roll/Pitch heranziehen und den klassischen Stabilisierungs-PD schrittweise durch Netzwerk-Ausgabe ersetzen oder parallel als Sicherheitsnetz (Regler greift nur bei Grenzwertüberschreitung ein) laufen lassen.

**Phase C (Stretch, optional):** Biologisch stärker fundierte Variante mit reiner Flügel/Halteren-Pool-Decodierung ohne Beimischung der Bein-Pools, eventuell mit einem Flapping-Wing-/Ornithopter-Modell statt Quadrocopter — das ist aber ein eigenes, deutlich größeres Forschungsthema und für dieses Projekt nur als "könnte man später" vermerkt, nicht eingeplant.

Decoder-Formel (Spike-Rate → normierter Wert) kann 1:1 aus dem Originalprojekt übernommen werden: `value = 1 − exp(−(rate − baseline) / 40 Hz)`, Gamma/Master-Skalierung wie dort, nur die Ziel-Größe ändert sich von "Servo-Mikrosekunden" zu "RPM-Offset um einen Hover-Basiswert".

## 6. Feedback-Loop (Propriozeption)

Die simulierte IMU (Gyro + Beschleunigung) aus gym-pybullet-drones speist die Balance-/Propriozeptions-Neuronen — an derselben Stelle, an der beim Strandbeest reale Gyro-/Servo-Geschwindigkeitsdaten ankommen. Das ist biologisch sogar stimmiger als beim Laufroboter: Halteren sind im echten Tier der Fluglage-Sensor, kein Bein-Sensor.

## 7. Phasenplan / Meilensteine

1. **M0 – Setup:** gym-pybullet-drones installieren, headless zum Laufen bringen, manuelles Hover per festen RPM-Werten testen (reine Sanity-Check, kein Konnektom beteiligt).
2. **M1 – Offline-Kopplung:** Kamerabild aus der Simulation aufzeichnen, offline durch den bestehenden Medulla-Encoder + Konnektom-Sim schicken, Motor-Neuron-Output loggen. Noch kein geschlossener Kreis — nur prüfen, ob aus synthetischen Kamerabildern überhaupt plausible Aktivität herauskommt.
3. **M2 – Geschlossener Kreis, Phase A (2 DOF):** Sum/Differenz-Mapping wie oben, PD-Regler für Roll/Pitch, Ziel: Drohne bleibt stabil in der Luft und reagiert nachvollziehbar (z. B. Gier-Drift) auf visuelle Reize.
4. **M3 – Tuning:** Gain-Kalibrierung zwischen Spike-Rate und RPM-Offset, Baseline-Justierung, Stabilitätstests (verschiedene Startbedingungen, Störungen).
5. **M4 – Phase B (4 DOF):** Roll/Pitch schrittweise vom Netzwerk übernehmen lassen.
6. **M5 (Stretch):** Auswertung, ob Verhalten wie Looming-Ausweichen oder Höhenhaltung über optischen Fluss emergent auftritt — das wäre das eigentliche wissenschaftlich interessante Ergebnis.

Bewusst kein Meilenstein "auf echte Hardware übertragen" in diesem Plan — das ist explizit ausgeklammert, bis die Simulation zeigt, dass es überhaupt sinnvoll fliegt.

## 8. Rechenleistung

GeNN-GPU-Backend läuft über CUDA — deine 3× M40 (Maxwell, Compute Capability 5.2) werden unterstützt, auch wenn nicht mehr aktuell. Der im Originalprojekt genannte Echtzeitfaktor (0,23× CPU/Numba, 1,46× GPU/GeNN) ist für reine Simulation unkritisch, da Physik- und Neuronen-Schritt im Lockstep statt in Wanduhrzeit laufen können — die Kopplung muss also nicht "echtzeitfähig" sein, nur konsistent getaktet (z. B. 200 Kernel-Schritte Konnektom pro 20-ms-I/O-Zyklus, wie im Original).

## 9. Ausgangscode

Zwei Repos aus der Recherche kommen als Fork-Basis infrage:

- [DOOMFLY](https://github.com/nftechie/doomfly) — MIT-Lizenz, am freizügigsten.
- [eonsystems/fly-brain](https://github.com/eonsystemspbc/fly-brain) — GPL-2.

Falls dir – wie beim Mold-Generator-Projekt – wichtig ist, dass Weiterentwicklungen quelloffen bleiben, wäre GPL-2 die konsequentere Wahl; falls du dir spätere kommerzielle Optionen offenhalten willst, MIT. Reine Geschmacksfrage, technisch unterscheiden sich beide nicht in der Encoder/Decoder-Architektur.

## 10. Offene Risiken

- Kamera-Rendering in PyBullet könnte für den optischen Fluss zu verrauscht/niedrig aufgelöst sein — ggf. Nachbearbeitung (Weichzeichnung, Kontrastanpassung) nötig, um dem Verhalten einer echten Facettenaugen-Optik näherzukommen.
- Die 815-Motor-Neuronen-Pools sind nicht sauber nach "vorne/hinten" trennbar (nur links/rechts ist biologisch eindeutig) — Phase B braucht möglicherweise Kompromisse bei der Poolauswahl.
- Ohne Sicherheitsregler (Phase A/B-Übergang) kann die Netzwerkausgabe instabile RPM-Sprünge erzeugen — in Simulation ungefährlich, aber ein Grund, real fliegende Hardware in diesem Projekt konsequent außen vor zu lassen.

## 11. Nächster konkreter Schritt

M0: `gym-pybullet-drones` lokal installieren, ein Hover-Beispiel mit `ActionType.RPM` headless zum Laufen bringen, und einmal `ObservationType.RGB` auslesen, um zu sehen, wie ein Frame tatsächlich aussieht, bevor es an den Medulla-Encoder geht.
