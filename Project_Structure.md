# Project Structure

```text
Race-Strategy-Decision-Support-System/
│
├── app/
│   ├── pages/
│   ├── assets/
│   └── app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── docs/
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── ml/
│   ├── optimization/
│   ├── simulation/
│   ├── visualization/
│   └── utils/
│
├── tests/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── roadmap.md
```

---

# Folder Explanations

## app/

Contains the final application that the user interacts with.

Example contents:

* Streamlit application
* Dashboard pages
* User interface
* Graphs
* Strategy recommendation screen

Example:

```
app/
    app.py
    pages/
        Strategy.py
        Data_Explorer.py
        Model_Analysis.py
```

---

## data/

Contains all datasets.

### raw/

Original downloaded datasets.

Never edit these files.

Examples:

* Bahrain 2023
* Monaco 2024
* Silverstone 2025

These serve as your permanent source of truth.

---

### processed/

Cleaned datasets generated from the raw data.

Examples:

* Missing values removed
* Columns standardized
* Feature engineering completed

These are the datasets your models will use.

---

### external/

Optional data from other sources.

Examples:

* Weather
* Circuit information
* Tire compound information
* Team information

---

## docs/

Contains project documentation.

Examples:

* Project report
* Architecture diagrams
* UML diagrams
* Research notes
* Design decisions
* Images used in the README

Think of this as the project's engineering notebook.

---

## models/

Stores trained machine learning models.

Examples:

```
random_forest.pkl

xgboost.json

lap_time_model.pkl

tire_model.pkl
```

This folder stores the trained models, **not** the code that trains them.

---

## notebooks/

Jupyter notebooks used for experimentation.

Examples:

```
01_data_exploration.ipynb

02_feature_engineering.ipynb

03_model_training.ipynb

04_model_evaluation.ipynb
```

Use notebooks to explore ideas, visualize data, and test models.

Once something works, move the final implementation into the `src/` directory.

---

# src/

Contains the project's actual source code.

Everything here should be reusable and organized into modules.

---

## src/data/

Responsible for loading and preparing data.

Examples:

* Download race data
* Load CSV files
* Clean datasets
* Merge datasets

---

## src/features/

Feature engineering.

Examples:

Create variables such as:

* Tire age
* Average lap pace
* Tire degradation trend
* Position change
* Gap trend

---

## src/ml/

Machine learning code.

Examples:

* Train models
* Save models
* Load models
* Evaluate accuracy
* Make predictions

---

## src/optimization/

The heart of the project.

Contains the pit strategy optimizer.

Responsibilities include:

* Simulate every possible pit lap
* Compare strategies
* Compute expected finishing position
* Recommend the optimal strategy

---

## src/simulation/

Race simulation.

Responsibilities include:

* Simulate race progression
* Predict future laps
* Apply tire degradation
* Simulate pit stops
* Update race positions

Future versions may include:

* Safety Cars
* Rain
* Virtual Safety Cars

---

## src/visualization/

Generates charts.

Examples:

* Tire wear curves
* Lap time plots
* Strategy comparison graphs
* Position evolution

---

## src/utils/

Utility functions used throughout the project.

Examples:

* Configuration
* Constants
* Helper functions
* Logging
* File handling

---

## tests/

Contains automated tests.

Examples:

* Test tire degradation calculations
* Test optimization logic
* Test data loading
* Test prediction functions

Writing tests helps ensure that new changes don't accidentally break existing functionality.

---

# Root Files

## README.md

The project's homepage on GitHub.

Should include:

* Project description
* Installation instructions
* Features
* Screenshots
* Architecture
* Roadmap
* Results
* Future work

---

## requirements.txt

Lists all required Python packages.

Example:

```
pandas
numpy
scikit-learn
xgboost
streamlit
plotly
fastf1
```

---

## .gitignore

Specifies files Git should ignore.

Examples:

* Python cache
* Virtual environments
* Large datasets
* Temporary files

---

## LICENSE

Contains the project's software license.

Recommended:

MIT License

---

## roadmap.md

Tracks the project's progress.

Example:

* ✅ Collect race data
* ⬜ Clean datasets
* ⬜ Train tire degradation model
* ⬜ Build optimizer
* ⬜ Develop dashboard
* ⬜ Validate against historical races
* ⬜ Publish Version 1.0

```
```
