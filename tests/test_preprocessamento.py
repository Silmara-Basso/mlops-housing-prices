# Automação das Operações de CI/CD no Pipeline de Machine Learning e IA

# Imports
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
from pipeline.chp_preprocessa_dados import chp_preprocessa_dados

@pytest.fixture
def sample_data():
    data = {
        'longitude': [-122.23, -122.22, -122.24, -122.25],
        'latitude': [37.88, 37.86, 37.85, 37.84],
        'housing_median_age': [41, 21, 52, 52],
        'total_rooms': [880, 7099, 1467, 1274],
        'total_bedrooms': [129, 1106, 190, 235],
        'population': [322, 2401, 496, 558],
        'households': [126, 1138, 177, 219],
        'median_income': [8.3252, 8.3014, 7.2574, 5.6431]
    }
    return pd.DataFrame(data)

def test_preprocess_data(sample_data):
    
    df_cleaned, _, scaler = chp_preprocessa_dados(sample_data)
    
    assert df_cleaned.isnull().sum().sum() == 0  
    assert scaler is not None  
    assert df_cleaned.shape[1] == sample_data.shape[1]  
