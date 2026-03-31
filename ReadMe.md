# Deep Learning Diabetes Prediction

## Project Description
This project aims to predict diabetes using deep learning techniques. The dataset used is derived from health indicators collected in the BRFSS 2015 survey. The project includes data preprocessing, model training, and evaluation, with deliverables provided as Jupyter notebooks.

## Project Structure
The project is organized as follows:

```
.
├── datasets/
│   ├── diabetes_012_health_indicators_BRFSS2015.csv  # Raw dataset
│   ├── processed_X.csv                               # Preprocessed features
│   └── processed_y.csv                               # Preprocessed labels
├── deliverables/
│   ├── deliverable1.ipynb                           # Notebook for data preprocessing
│   └── deliverable2.ipynb                           # Notebook for model training and evaluation
├── diagrams/                                        # Placeholder for diagrams
│   ├── deliverable1/                                # Diagrams for deliverable 1
│   └── deliverable2/                                # Diagrams for deliverable 2
├── models/                                          # Placeholder for saved models
├── ReadMe.md                                        # Project documentation
├── Requirements.txt                                 # Python dependencies
```

## Setup Instructions
Follow these steps to set up the project environment and run the notebooks:

### 1. Create a Virtual Environment
1. Open a terminal and navigate to the project directory.
2. Run the following command to create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

### 2. Install Dependencies
With the virtual environment activated, install the required dependencies:
```bash
pip install -r Requirements.txt
```

### 3. Run the Notebooks
1. Ensure the virtual environment is activated.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open the desired notebook from the `deliverables/` folder and run the cells.

## Dataset
The dataset used in this project is `diabetes_012_health_indicators_BRFSS2015.csv`, which contains health indicators for diabetes prediction. Preprocessed versions of the dataset are stored as `processed_X.csv` and `processed_y.csv` in the `datasets/` folder.

## Deliverables
- **deliverable1.ipynb**: Covers data preprocessing steps, including cleaning and feature engineering.
- **deliverable2.ipynb**: Focuses on model training, evaluation, and results visualization.

## Contact
For any questions or contributions, please contact the project team.