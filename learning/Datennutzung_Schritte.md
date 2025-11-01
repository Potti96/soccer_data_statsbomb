# Datennutzung mit Fußballdaten - Lernpfad für Python

Dieses Dokument beschreibt die wichtigsten Schritte der Datennutzung für ein Python-Projekt mit Fußballdaten, insbesondere mit Fokus auf versicherungsrelevante Analysen.

## 1. Datenanbindung

### 1.1 Datenquellen identifizieren
- **StatsBomb Open Data**: Kostenlose Fußballdaten (Events, Tracking-Daten)
- **APIs**: RESTful APIs für Live- oder historische Daten
- **CSV/JSON Files**: Lokale oder externe Dateien
- **Datenbanken**: SQL (PostgreSQL, MySQL) oder NoSQL (MongoDB)

### 1.2 Python-Bibliotheken für Datenanbindung
```python
# Bibliotheken importieren
import pandas as pd          # Datenmanipulation
import requests             # API-Aufrufe
import json                 # JSON-Verarbeitung
import sqlite3              # SQL-Datenbanken
from pathlib import Path    # Dateipfade
```

### 1.3 Praktische Beispiele
- **CSV-Dateien laden**: `df = pd.read_csv('file.csv')`
- **JSON-Dateien laden**: `df = pd.read_json('file.json')`
- **API-Aufrufe**: `response = requests.get(url)`
- **Datenbankverbindung**: `conn = sqlite3.connect('database.db')`

### 1.4 Datenqualität prüfen
- Dateigröße und Struktur verstehen
- Encoding prüfen (UTF-8, ISO-8859-1)
- Authentifizierung bei APIs sicherstellen
- Fehlerbehandlung implementieren

---

## 2. Datenexploration

### 2.1 Erste Übersicht über die Daten
- **Dimensionen verstehen**: `df.shape` (Zeilen, Spalten)
- **Spaltennamen**: `df.columns`
- **Datentypen**: `df.dtypes`
- **Erste Zeilen**: `df.head()`, `df.tail()`
- **Grundstatistiken**: `df.describe()`

### 2.2 Fehlende Werte identifizieren
- `df.isnull().sum()` - Fehlende Werte pro Spalte
- `df.isnull().sum().sum()` - Gesamtzahl fehlender Werte
- Visualisierung mit Heatmaps

### 2.3 Datenverteilungen analysieren
- **Kategorische Variablen**: `df['column'].value_counts()`
- **Numerische Variablen**: Histogramme, Boxplots
- **Korrelationen**: `df.corr()`

### 2.4 Python-Bibliotheken für Exploration
```python
import matplotlib.pyplot as plt  # Visualisierung
import seaborn as sns            # Erweiterte Visualisierung
import numpy as np               # Numerische Operationen
```

### 2.5 Wichtige Visualisierungen
- Histogramme für Verteilungen
- Scatter Plots für Zusammenhänge
- Boxplots für Ausreißer
- Heatmaps für Korrelationen
- Time Series Plots für zeitliche Entwicklungen

---

## 3. Datenaufbereitung

### 3.1 Fehlende Werte behandeln
- **Identifikation**: Wo fehlen Daten?
- **Strategien**:
  - Löschen: `df.dropna()`
  - Ersetzen mit Mittelwert/Median: `df.fillna(df.mean())`
  - Forward/Backward Fill: `df.fillna(method='ffill')`
  - Interpolation für Zeitreihen

### 3.2 Duplikate entfernen
- `df.duplicated()` - Duplikate identifizieren
- `df.drop_duplicates()` - Duplikate entfernen

### 3.3 Datenbereinigung
- **Outlier-Behandlung**: IQR-Methode, Z-Score
- **Inkonsistente Formate**: Datumsformatierung, Textnormalisierung
- **Falsche Datentypen korrigieren**: `pd.to_datetime()`, `pd.to_numeric()`

### 3.4 Feature Engineering
- **Neue Variablen erstellen**:
  - Tore pro Spiel
  - Passquote
  - Ballbesitz
  - Verletzungsdauer (für Versicherungsrelevanz)
- **Kategorische Variablen codieren**:
  - One-Hot Encoding: `pd.get_dummies()`
  - Label Encoding
- **Zeitbasierte Features**:
  - Jahreszeiten, Monate
  - Spieltage
  - Wochentage

### 3.5 Daten-Transformation
- **Normalisierung**: Min-Max Scaling, Z-Score Normalization
- **Log-Transformation**: Für schiefe Verteilungen
- **Aggregation**: Gruppierung nach Teams, Spielern, Zeiträumen

---

## 4. Data Science Modelle / Analysen

### 4.1 Versicherungsrelevante Analysen

#### 4.1.1 Musteranalyse (Pattern Analysis)
- **Verletzungsmuster erkennen**:
  - Welche Spieler/Positionen haben häufiger Verletzungen?
  - Zeitliche Muster (Saisonende, nach vielen Spielen)
  - Zusammenhang mit Spielintensität
- **Python-Ansätze**:
  - Clustering (K-Means, DBSCAN) für Spieler-Segmentierung
  - Association Rules für häufige Verletzungsmuster
  - Zeitreihenanalyse für saisonale Muster

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# Clustering-Beispiel für Risikogruppen
```

#### 4.1.2 Risikobewertung (Risk Assessment)
- **Vorhersagemodelle für Verletzungen**:
  - Klassifikationsmodelle (Logistic Regression, Random Forest)
  - Welche Spieler haben hohes Verletzungsrisiko?
  - Faktoren: Alter, Position, Spielminuten, Historie
- **Risiko-Scoring**:
  - Score-Modell entwickeln
  - Risikokategorien (niedrig, mittel, hoch)

#### 4.1.3 Präventive Analysen
- **Early Warning Indicators**:
  - Abnahme der Performance als Warnsignal
  - Erhöhte Belastung vor Verletzung
  - Änderungen in Spielmustern

### 4.2 Wichtige Machine Learning Modelle

#### 4.2.1 Überwachtes Lernen (Supervised Learning)
- **Klassifikation**:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting (XGBoost, LightGBM)
  - SVM (Support Vector Machines)
- **Regression**:
  - Lineare Regression
  - Random Forest Regressor
  - Für Vorhersagen (z.B. erwartete Tore)

#### 4.2.2 Unüberwachtes Lernen (Unsupervised Learning)
- **Clustering**:
  - K-Means
  - Hierarchical Clustering
  - DBSCAN
- **Dimensionalitätsreduktion**:
  - PCA (Principal Component Analysis)
  - t-SNE

#### 4.2.3 Zeitreihenanalyse
- ARIMA-Modelle
- Prophet (Facebook)
- LSTM (Long Short-Term Memory) für komplexe Muster

### 4.3 Modellevaluierung
- **Metriken für Klassifikation**:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC Curve
  - Confusion Matrix
- **Metriken für Regression**:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score
- **Cross-Validation**: K-Fold Cross-Validation zur robusten Bewertung

### 4.4 Python-Bibliotheken für ML
```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
```

---

## 5. Zusätzliche wichtige Schritte

### 5.1 Datenvisualisierung
- **Dashboard-Erstellung**:
  - Plotly für interaktive Visualisierungen
  - Matplotlib/Seaborn für statische Plots
- **Reporting**: Automatisierte Reports generieren

### 5.2 Datenvalidierung
- Plausibilitätschecks
- Business Logic Validierung
- Datenqualitätsmetriken

### 5.3 Reproduzierbarkeit
- **Version Control**: Git für Code-Versionierung
- **Dependencies**: `requirements.txt` für Pakete
- **Dokumentation**: Code-Kommentare, README

### 5.4 Performance-Optimierung
- Effiziente Datenstrukturen wählen
- Vektorisierung statt Loops (NumPy, Pandas)
- Parallelisierung wo möglich

---

## 6. Versicherungsspezifische Anwendungsfälle

### 6.1 Prämienkalkulation
- Risiko-basierte Prämien für Spieler-Versicherungen
- Dynamische Anpassung basierend auf Performance/Verletzungen

### 6.2 Schadensvorhersage
- Wahrscheinlichkeit von Verletzungen
- Erwartete Kosten pro Verletzungstyp
- ROI-Analyse für Präventionsmaßnahmen

### 6.3 Portfolio-Management
- Risikodiversifikation über verschiedene Teams/Ligen
- Stress-Testing für extreme Szenarien

### 6.4 Compliance & Reporting
- Regulatorische Berichte
- Risiko-Dashboards für Management
- Automatisierte Alerts bei kritischen Mustern

---

## 7. Lernressourcen & Weiteres Vorgehen

### 7.1 Praktische Übungen
1. StatsBomb Open Data herunterladen und explorieren
2. Verletzungsdaten mit Spielerperformance korrelieren
3. Ein einfaches Risiko-Score-Modell entwickeln
4. Visualisierungen für verschiedene Stakeholder erstellen

### 7.2 Wichtige Python-Bibliotheken im Überblick
- **Datenmanipulation**: pandas, numpy
- **Visualisierung**: matplotlib, seaborn, plotly
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Statistik**: scipy, statsmodels
- **Zeitreihen**: prophet, statsmodels

### 7.3 Best Practices
- Code strukturieren in Funktionen/Klassen
- Modularität fördern
- Testing (Unit Tests) wo sinnvoll
- Clean Code Prinzipien beachten

---

## Zusammenfassung der Schritte

1. ✅ **Datenanbindung**: Daten aus verschiedenen Quellen laden
2. ✅ **Datenexploration**: Daten verstehen und erste Insights gewinnen
3. ✅ **Datenaufbereitung**: Daten bereinigen und für Analyse vorbereiten
4. ✅ **Data Science Modelle**: Muster erkennen, Vorhersagen treffen, Risiken bewerten
5. **Visualisierung**: Ergebnisse kommunizieren
6. **Validierung & Testing**: Modelle und Ergebnisse validieren
7. **Deployment**: Modelle produktiv nutzen (optional für Lernprojekt)

---

*Dieses Dokument dient als Leitfaden für den Lernprozess mit Fußballdaten und fokussiert auf versicherungsrelevante Anwendungsfälle.*

