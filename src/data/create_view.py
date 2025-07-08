from google.cloud import bigquery
import os
import logging
from enum import Enum
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ViewType(Enum):
    """Enum for different view implementation types."""
    STANDARD = "standard"
    OPTIMIZED = "optimized"
    MATERIALIZED = "materialized"

def create_games_features_view(
    client: bigquery.Client, 
    dataset_id: str,
    view_type: ViewType = ViewType.OPTIMIZED
):
    """
    Create or replace the games features view using the specified implementation.
    
    Args:
        client (bigquery.Client): Authenticated BigQuery client
        dataset_id (str): ID of the dataset where the view will be created
        view_type (ViewType, optional): Type of view to create. Defaults to ViewType.OPTIMIZED.
    """
    # Determine which SQL file to use
    if view_type == ViewType.STANDARD:
        sql_filename = 'games_features_view.sql'
        logger.info("Creating standard view (warning: may have performance issues)")
    elif view_type == ViewType.OPTIMIZED:
        sql_filename = 'games_features_optimized_view.sql'
        logger.info("Creating optimized view")
    elif view_type == ViewType.MATERIALIZED:
        sql_filename = 'games_features_materialized_view.sql'
        logger.info("Creating materialized view (best performance)")
    else:
        raise ValueError(f"Invalid view type: {view_type}")
    
    # Read the SQL template
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sql_path = os.path.join(current_dir, sql_filename)
    
    with open(sql_path, 'r') as f:
        view_template = f.read()
    
    # Replace the dataset placeholder
    view_sql = view_template.replace('{dataset}', dataset_id)
    
    try:
        # Log the SQL file being used
        logger.info(f"Using SQL template from: {sql_filename}")
        
        # Create the view
        job_config = bigquery.QueryJobConfig()
        query_job = client.query(view_sql, job_config=job_config)
        
        # Wait for the job to complete and log details
        query_result = query_job.result()
        
        logger.info(f"View created successfully in dataset {dataset_id} using {view_type.value} implementation")
        
        # Log additional job details
        logger.info(f"Job ID: {query_job.job_id}")
        logger.info(f"Total bytes processed: {query_job.total_bytes_processed}")
        
    except Exception as e:
        logger.error(f"Error creating view: {e}")
        
        # If it's a BigQuery specific error, provide more context
        if hasattr(e, 'errors'):
            for error in e.errors:
                logger.error(f"BigQuery Error: {error}")
        
        # Re-raise the exception to stop execution
        raise

def refresh_materialized_view(
    client: bigquery.Client,
    dataset_id: str
):
    """
    Refresh the materialized view by calling the stored procedure.
    
    Args:
        client (bigquery.Client): Authenticated BigQuery client
        dataset_id (str): ID of the dataset where the view exists
    """
    try:
        # Call the refresh procedure
        refresh_sql = f"CALL `{dataset_id}.refresh_games_features_materialized`()"
        logger.info(f"Refreshing materialized view with: {refresh_sql}")
        
        query_job = client.query(refresh_sql)
        query_result = query_job.result()
        
        logger.info("Materialized view refreshed successfully")
        
    except Exception as e:
        logger.error(f"Error refreshing materialized view: {e}")
        
        # If it's a BigQuery specific error, provide more context
        if hasattr(e, 'errors'):
            for error in e.errors:
                logger.error(f"BigQuery Error: {error}")
        
        # Re-raise the exception to stop execution
        raise

def main():
    """
    Main function to demonstrate creating the games features view.
    Assumes default credentials or environment variables are set up.
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create or refresh the games features view')
    parser.add_argument('--type', choices=['standard', 'optimized', 'materialized'], 
                        default='materialized',
                        help='Type of view to create (default: optimized)')
    parser.add_argument('--refresh', action='store_true',
                        help='Refresh the materialized view instead of creating it')
    args = parser.parse_args()
    
    # Create BigQuery client
    client = bigquery.Client()
    
    # Get dataset from environment or configuration
    dataset_id = os.environ.get('BGG_DATASET', 'bgg_data_dev')
    
    if args.refresh:
        # Refresh the materialized view
        refresh_materialized_view(client, dataset_id)
    else:
        # Create the view with the specified type
        view_type = ViewType(args.type)
        create_games_features_view(client, dataset_id, view_type)

if __name__ == '__main__':
    main()
