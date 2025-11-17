from google.cloud import bigquery
from google.cloud import bigquery_datatransfer
import os
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ViewType(Enum):
    """Enum for materialized view implementation."""

    MATERIALIZED = "materialized"


def schedule_view_refresh(
    project_id: str,
    dataset_id: str,
    schedule: str = "every day 07:30",
) -> None:
    """
    Schedule automatic refresh of the materialized view.

    Args:
        project_id (str): GCP project ID
        dataset_id (str): BigQuery dataset ID
        schedule (str): Schedule in cron format or using 'every X hours/days'
    """
    transfer_client = bigquery_datatransfer.DataTransferServiceClient()
    parent = transfer_client.common_project_path(project_id)

    transfer_config = {
        "display_name": "Refresh games features materialized view",
        "data_source_id": "scheduled_query",
        "destination_dataset_id": dataset_id,
        "params": {
            "query": f"CALL `{project_id}.{dataset_id}.refresh_games_features_materialized`();"
        },
        "schedule": schedule,
    }

    try:
        transfer_client.create_transfer_config(
            parent=parent, transfer_config=transfer_config
        )
        logger.info(f"Successfully scheduled view refresh: {schedule}")
    except Exception as e:
        logger.error(f"Error scheduling view refresh: {e}")
        raise


def create_games_features_view(
    client: bigquery.Client,
    dataset_id: str,
    view_type: ViewType = ViewType.MATERIALIZED,
):
    """
    Create or replace the games features view using the specified implementation.

    Args:
        client (bigquery.Client): Authenticated BigQuery client
        dataset_id (str): ID of the dataset where the view will be created
        view_type (ViewType, optional): Type of view to create. Defaults to ViewType.OPTIMIZED.
    """
    # Use materialized view SQL
    sql_filename = "games_features_materialized_view.sql"
    logger.info("Creating materialized view (best performance)")

    # Read the SQL template
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sql_path = os.path.join(current_dir, sql_filename)

    with open(sql_path, "r") as f:
        view_template = f.read()

    # Replace the dataset placeholder
    view_sql = view_template.replace("{dataset}", dataset_id)

    try:
        # Log the SQL file being used
        logger.info(f"Using SQL template from: {sql_filename}")

        # Create the materialized view using the full SQL template
        job_config = bigquery.QueryJobConfig()
        query_job = client.query(view_sql, job_config=job_config)
        query_job.result()  # Wait for the job to complete

        table_id = f"{dataset_id}.games_features_materialized"
        logger.info(f"Materialized view created successfully: {table_id}")

    except Exception as e:
        logger.error(f"Error creating materialized view: {e}")

        # If it's a BigQuery specific error, provide more context
        if hasattr(e, "errors"):
            for error in e.errors:
                logger.error(f"BigQuery Error: {error}")

        # Re-raise the exception to stop execution
        raise


def main():
    """
    Main function to demonstrate creating the games features view.
    Uses configuration from config.yaml
    """
    import argparse
    from ..utils.config import load_config

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create the games features materialized view"
    )
    parser.add_argument(
        "--type",
        choices=["materialized"],
        default="materialized",
        help="Type of view to create (materialized)",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="every day 07:30",
        help="Schedule for view refresh (default: every day 07:30)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Get BigQuery configuration for current environment
    bigquery_config = config.get_bigquery_config()

    # Create BigQuery client using configuration
    client = bigquery_config.get_client()

    # Use dataset from configuration
    dataset_id = bigquery_config.dataset

    # Create the view with the specified type
    view_type = ViewType(args.type)
    create_games_features_view(client, dataset_id, view_type)

    # Schedule the view refresh
    schedule_view_refresh(
        project_id=bigquery_config.project_id,
        dataset_id=dataset_id,
        schedule=args.schedule,
    )


if __name__ == "__main__":
    main()
