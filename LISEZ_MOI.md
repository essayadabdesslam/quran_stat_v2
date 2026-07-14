# Outil d'analyse lexicale du Coran — Guide d'installation

## Avertissement méthodologique

Cet outil est **purement descriptif** : fréquences lexicales, concordance,
similarité statistique de surface (TF-IDF). Il ne produit **aucune**
interprétation théologique et ne remplace en aucun cas l'exégèse (tafsir)
réalisée par des spécialistes qualifiés. Toute utilisation interprétative
des résultats doit être validée par des personnes compétentes (ulémas,
linguistes arabisants, islamologues).

## Contenu du dossier

```
quran_app/
├── app.py                  # Application Streamlit (interface en arabe)
├── requirements.txt        # Dépendances Python
├── data/
│   └── quran_data.json     # Texte coranique complet (114 sourates, 6236 versets, texte uthmani)
└── fonts/
    ├── Amiri-Regular.ttf   # Police arabe (licence libre SIL OFL) pour le PDF
    └── Amiri-Bold.ttf
```

## Installation

```bash
cd quran_app
python3 -m venv venv
source venv/bin/activate        # sous Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre dans le navigateur à l'adresse `http://localhost:8501`.
**L'interface est entièrement en arabe** (texte, boutons, onglets), avec
mise en page adaptée de droite à gauche (RTL).

## Fonctionnalités

1. **نطاق التحليل (Portée de l'analyse, barre latérale)** — choisir entre
   l'ensemble du Coran ou une sélection spécifique d'une ou plusieurs sourates.
2. **📚 النص** — affichage du texte des sourates sélectionnées.
3. **🔎 البحث والتوافق** — recherche d'un mot et concordance dans la portée choisie.
4. **📊 التكرار المعجمي** — statistiques de fréquence des mots.
5. **🧭 التشابه النصي** — versets les plus proches lexicalement d'un verset
   de référence (option pour chercher dans tout le Coran ou seulement dans
   la portée sélectionnée).
6. **📄 تقرير PDF** — génère un rapport PDF (texte arabe correctement
   rendu, police Amiri) reprenant fréquences, résultats de recherche et
   similarité, téléchargeable directement depuis l'application.

## Source du texte coranique

Le fichier `data/quran_data.json` contient le texte uthmani du Coran
(114 sourates, 6236 versets), issu d'un dépôt open source largement utilisé
par la communauté de développeurs (projet `quran-json`, texte basé sur
*The Noble Qur'an Encyclopedia*). Pour un usage en production, il est
recommandé de vérifier ce texte par recoupement avec une source de
référence (par exemple Tanzil.net) et de faire valider l'intégrité du
corpus par des personnes compétentes.

## Limites à garder à l'esprit

- La "similarité" mesurée est purement lexicale (mots partagés), pas
  thématique ni doctrinale.
- Le script de normalisation (suppression des diacritiques) peut, dans de
  rares cas, fusionner des mots orthographiquement proches mais
  sémantiquement distincts.
- Aucune fonctionnalité de cet outil ne doit être présentée comme une
  interprétation ou un jugement religieux.
