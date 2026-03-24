# Projectvoorstel: Computationele Authenticatie van Kunst
**Wiskundige vingerafdrukken als bewijs van echtheid**

| | |
|---|---|
| **Duur** | 5 weken |
| **Profiel** | Toegepaste wiskunde / numerieke methoden |
| **Domein** | Kunstanalyse / Data-analyse / Statistiek |
| **Potentieel** | Direct te vertalen naar commercieel product |

---

## 1. Achtergrond & Probleemstelling

Kunstfraude is wereldwijd een miljardenproblemen. Vervalsers zijn in staat de visuele stijl van een kunstenaar overtuigend na te bootsen, maar reproduceren zelden de statistische en wiskundige structuur die een authentiek werk kenmerkt.

Een kunstwerk laat onbewuste wiskundige vingerafdrukken achter: in de manier waarop verf wordt aangebracht, hoe kleurovergangen verlopen, en welke fractale structuren ontstaan in de textuur. Deze eigenschappen zijn stijlgebonden en extreem moeilijk te vervalsen.

**Centrale vraag:** Welke wiskundige kenmerken van een schilderij zijn uniek genoeg om als authenticatiebewijs te dienen?

---

## 2. Drie wiskundige invalshoeken

### Fractale dimensie-analyse
- Box-counting dimensie op meerdere ruimteschalen
- Penseeltextuur & chaos-structuur

### Fourier- & wavelet-analyse
- Frequentiespectrum per kleurkanaal op hoge resolutie
- Techniek-specifieke textuurhandtekening

### Kleurgradiënt statistiek
- Verdeling van lokale gradiënten over het hele werk
- Robuust tegen vrijhand-vervalsingen

---

### Fractale dimensie-analyse

Echte schilderijen vertonen karakteristieke fractale eigenschappen in penseelstreken en verfstructuur. Jackson Pollock werd al succesvol geauthenticeerd via fractale dimensieanalyse. De wiskundige kern: bereken de box-counting dimensie op verschillende schalen en vergelijk met bekende werken van dezelfde kunstenaar.

### Fourier- en wavelet-analyse van textuur

Digitaliseer een schilderij op hoge resolutie en analyseer het frequentiespectrum per kleurkanaal. Elke kunstenaar heeft door zijn of haar specifieke techniek een karakteristiek textuurspectrum. De uitdaging zit in het onderscheiden van stijlgebonden vs. onderwerp-gebonden frequentiekenmerken.

### Statistische analyse van kleurgradiënten

Analyseer de verdeling van lokale kleurgradiënten over het hele werk. Dit is robuust tegen reproductie: een vervalser die pixel-voor-pixel kopieert reproduceert de gradiëntverdeling, maar een vrijhand-vervalser doet dat niet. Dit maakt de methode bijzonder praktisch inzetbaar.

---

## 3. Projectplanning (5 weken)

| Week | Focus |
|------|-------|
| 1 | Dataset samenstellen: gedigitaliseerde werken van 2–3 kunstenaars + bekende vervalsingen |
| 2 | Feature-extractie: implementeer fractale dimensie-analyse en wavelet-transformatie |
| 3 | Statistische modellering: classificatiemodel bouwen op geëxtraheerde features |
| 4 | Validatie op testcases, foutanalyse en drempelwaardebepaling |
| 5 | Technisch rapport, demo-tool en teampresentatie |

---

## 4. Deliverables

**Python feature-extractie module**
Herbruikbare functies voor fractale dimensie, wavelet- en gradiëntanalyse

**Classificatiemodel**
Getraind en gevalideerd model met gedocumenteerde nauwkeurigheid per methode

**Demo-tool / prototype**
Simpele interface: upload afbeelding → ontvang authenticiteitscore

**Technisch rapport**
10–15 pagina's met wiskundige onderbouwing en commerciële aanbevelingen

---

## 5. Wiskundige uitdaging

- Fractale dimensieberekening vereist numerieke stabiliteit bij meerdere schaalniveaus
- Wavelet-keuze (Haar, Daubechies, Morlet) heeft direct invloed op de detectiegevoeligheid
- Classificatie op kleine datasets vereist kennis van regularisatie en cross-validatie
- Evaluatie van authenticiteit is inherent een probleem met asymmetrische foutkosten (type I vs. type II)

---

## 6. Commerciële relevantie

Dit project levert direct een prototype op voor een API die afbeeldingen van kunstwerken analyseert en een authenticiteitscore retourneert. Potentiële afnemers:

- Veilinghuizen (pre-sale verificatie)
- Verzekeringsmaatschappijen (waardebepaling en fraudedetectie)
- Musea en galeries (aankoop due diligence)
- Particuliere verzamelaars

Jongens van Techniek beschikt al over de AI- en data-infrastructuur. Dit project levert de wiskundige kern voor een nieuw product.

---

## 7. Aanbevolen startpunten

**Papers:**
- Taylor et al. (1999) — *Fractal Analysis for Art Authentication* (over Pollock)
- Johnson et al. (2008) — *Wavelet-based texture analysis for painting authentication*

**Dataset:**
- WikiArt (wikiart.org) — gedigitaliseerde werken van honderden kunstenaars

**Python libraries:**
- `PyWavelets` (pywt)
- `scipy.ndimage` — fractale analyse
- `scikit-learn` — classificatie

---

*Jongens van Techniek — Intern projectvoorstel*
