from src.data.sdss_query import get_sdss_data
from src.data.preprocessing import process_sdss_data
import os

def run_sdss_query():
    """Retrieve clean, usable data from the SDSS Skyserver complete with the
    features needed for model development"""
    df = get_sdss_data()
    df = process_sdss_data(df)
    save_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    save_path = os.path.join(save_directory, 'sdss_processed_data.csv')
    df.to_csv(save_path, index=False)

if __name__ == '__main__':
    run_sdss_query()