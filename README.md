# Smart Agriculture with Machine Learning 

An educational Python project that simulates a smart irrigation system for an agricultural field with eight zones: four growing maize and four growing rice. A decision tree predicts which zones require irrigation, while a seven-day simulation tracks water consumption and soil moisture conditions.

## Project Overview

The project explores how machine learning can support irrigation decisions under resource constraints. It combines a decision tree classifier, a simplified agricultural simulation, safety rules, and performance metrics to evaluate irrigation decisions.

The simulation models two crop types with different soil moisture dynamics and limits irrigation to four zones per day.

## How It Works

### 1. Dataset and Input Features

The model uses five dataset columns:

- `humedad_suelo`: soil moisture.
- `temperatura`: air temperature.
- `humedad_aire`: air humidity.
- `cultivo`: crop type, encoded as 0 for maize and 1 for rice.
- `regar`: binary target indicating whether irrigation should be activated according to the selected data source.

The project supports two data sources.

**Synthetic dataset — recommended for the educational demonstration**

The program generates a reproducible dataset with 1,200 samples by default. Irrigation labels are created using an explicit heuristic based on soil moisture, air temperature, air humidity, crop type, and random noise.

These labels represent the output of the defined heuristic, not experimentally measured agricultural irrigation requirements.

**Kaggle dataset — experimental mode**

The program can process `Smart_Farming_Crop_Yield_2024.csv`, filtering the data to retain the Maize and Rice crops.

In this mode, the `irrigation_type` column is converted into a binary target. A recorded irrigation type other than `None` is interpreted as irrigation having occurred.

This represents observed irrigation history, not whether irrigation was actually necessary. Rows with missing irrigation labels are excluded rather than incorrectly classified as instances without irrigation. The program displays warnings explaining these limitations.

### 2. Machine Learning Model

The project uses a `DecisionTreeClassifier` from scikit-learn.

The model receives four input features:

- Soil moisture.
- Air temperature.
- Air humidity.
- Crop type.

It predicts whether irrigation should be activated for each agricultural zone.

The model evaluation includes accuracy, a classification report, a confusion matrix, cross-validation scores when possible, and feature importance when a valid evaluation is available.

A reproducible train/test split and fixed random seeds are used. When the dataset does not support a valid evaluation, unavailable metrics are reported explicitly instead of presenting misleading results.

### 3. Seven-Day Irrigation Simulation

The simulated field contains eight zones:

| Zone group | Crop | Number of zones |
|---|---|---:|
| M1–M4 | Maize | 4 |
| A1–A4 | Rice | 4 |

Each simulated day follows these steps:

1. Generate daily weather conditions, including temperature, air humidity, and a 5% probability of rain.
2. Evaluate each zone using the trained decision tree.
3. Identify priority zones using an independent soil moisture safety rule.
4. Rank irrigation candidates while respecting the daily valve limit.
5. Activate the selected valves and update soil moisture.
6. Record water consumption and the percentage of healthy zones.

**Irrigation constraints and safety rules**

- A maximum of four zones can be irrigated per day.
- Each activated valve delivers 10 liters in the simulation.
- Zones below the configured soil moisture safety threshold are prioritized, even when the model does not request irrigation.
- If there are more candidates than available valves, priority zones are selected first, starting with the driest.
- The program reports priority zones that could not be irrigated because of the valve limit.

**Crop-specific moisture dynamics**

Maize and rice follow different configured moisture rules:

| Parameter | Maize | Rice |
|---|---:|---:|
| Moisture loss without irrigation | 4 percentage points/day | 6 percentage points/day |
| Moisture gain with irrigation | 14 percentage points | 16 percentage points |

These values are simulation parameters, not agronomically validated measurements. Rainfall adds a configured moisture increment, subject to the simulation's saturation rule.

### 4. Evaluation Criteria

The project defines a successful simulated week using three criteria:

| Metric | Target |
|---|---:|
| Total water consumption | 420 L or less |
| Water savings against the 560 L reference | At least 25% |
| Healthy zones | At least 80% every simulated day |

A zone is classified as healthy when its final daily soil moisture is at least 40%.

The evaluation checks the criteria separately and considers the week successful only when all of them are satisfied.

**Important:** the simulation's current valve constraint permits a maximum of 280 L over seven days (4 valves × 10 L × 7 days). Compared with the configured 560 L reference, this constraint alone imposes at least 50% apparent water savings. Therefore, total water savings cannot be attributed entirely to machine learning. The project distinguishes the savings imposed by the valve limit from additional differences in consumption.

### 5. Comparison Without Machine Learning

The program also compares the decision-tree irrigation strategy with a reference strategy that uses the soil moisture safety rule without machine learning.

Both strategies use the same simulated weather conditions and valve constraints for the comparison week.

The comparison reports differences in water consumption and soil moisture conditions. It helps explore the behavior of the two strategies under identical simulated conditions.

The results represent a single simulated week. They do not establish that machine learning is generally superior, nor do they demonstrate real-world water savings.

## Visualizations

The program generates five types of visualizations:

1. **Confusion matrix:** summarizes the classifier's predictions against the test labels when evaluation is valid.
2. **Soil moisture by zone:** tracks soil moisture throughout the simulation, including the initial state and the configured healthy moisture threshold.
3. **Daily water consumption:** shows daily water usage against the average daily target.
4. **Water consumption comparison:** compares the reference consumption, target consumption, and simulated consumption.
5. **Healthy zones per day:** tracks the proportion of zones meeting the configured soil moisture threshold.

Figures can be displayed and saved as PNG files.

### 1. Confusion Matrix
![Confusion Matrix](matriz%20de%20confusion%20ML.png)

### 2. Soil Moisture Evolution
![Soil Moisture](Evoluci%C3%B3n%20de%20humedad%20del%20suelo%20por%20zona%20ML.png)

### 3. Daily Water Consumption
![Daily Water Consumption](Consumo%20de%20agua%20diario%20durante%20la%20simulaci%C3%B3n%20ML.png)

### 4. Weekly Water Consumption
![Weekly Water Consumption](Consumo%20total%20semanal%20ML.png)

### 5. Healthy Zones
![Healthy Zones](%25%20de%20zonas%20saludables%20diarias%20ML.png)

## Data Quality and Reproducibility

The project includes:

- Dataset schema and value validation.
- Handling of missing or invalid dataset values.
- Source metadata for cached datasets.
- Fixed random seeds for reproducible dataset generation and simulation conditions.
- Safeguards against reporting invalid model evaluation metrics.
- Warnings when class distributions or labels make evaluation unreliable.

The program distinguishes between synthetic data and processed Kaggle data so their different meanings are not confused.

## Requirements

The program uses Python and the following libraries:

- NumPy
- pandas
- Matplotlib
- scikit-learn

## Running the Project

Run the default configuration, which attempts to use available data and falls back to the synthetic dataset when necessary:

```bash
python agricultura_inteligente.py
```

Force the recommended synthetic dataset:

```bash
python agricultura_inteligente.py --datos sintetico
```

Use the Kaggle dataset without falling back to synthetic data:

```bash
python agricultura_inteligente.py --datos kaggle
```

Save the generated plots to a directory named `graficos`:

```bash
python agricultura_inteligente.py --salida graficos
```

For Kaggle mode, download `Smart_Farming_Crop_Yield_2024.csv` and place it in the expected project directory.

## Limitations

This is an educational simulation, not a deployment-ready agricultural control system.

The simulation uses simplified weather generation, fixed moisture increments and decay rates, and configured thresholds. It does not model evapotranspiration, soil composition, drainage, real valve flow rates, or experimentally validated crop-specific water requirements.

The synthetic dataset is generated from a heuristic, meaning that good classification metrics may reflect how well the model learns that rule rather than how accurately it identifies genuine irrigation needs. The Kaggle mode has a separate limitation: its target represents recorded irrigation history rather than verified irrigation necessity.

Consequently, neither model metrics nor a successful simulated week should be interpreted as proof of real-world agricultural performance.

## Future Improvements

Potential improvements include:

- Testing the irrigation strategy across multiple random seeds and weather scenarios.
- Evaluating the model against more meaningful agronomic labels.
- Comparing strategies using repeated experiments and consistent metrics.
- Incorporating more realistic soil, crop, rainfall, and evapotranspiration models.
- Integrating physical sensors and irrigation hardware in a future implementation.
