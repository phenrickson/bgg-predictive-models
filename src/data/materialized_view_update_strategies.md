# Materialized View Update Strategies for Games Features

## Update Mechanisms

### 1. Manual Updates
```sql
-- Manually refresh the view
CALL `bgg_data_dev.refresh_games_features_materialized`();
```
- Developers can manually trigger the view refresh
- Useful for immediate updates after significant data changes

### 2. Scheduled Updates
#### A. BigQuery Scheduled Query
```bash
# Example gcloud command to create a scheduled query
bq mk --transfer_config \
  --project_id=your-project \
  --data_source=scheduled_query \
  --display_name="Daily Games Features Refresh" \
  --params='{"query":"CALL `bgg_data_dev.refresh_games_features_materialized`()"}' \
  --schedule="every 24 hours"
```
- Automatically refreshes the view at predefined intervals
- Configurable through Google Cloud Console or gcloud CLI
- Options include:
  - Daily refresh
  - Hourly refresh
  - Custom time intervals

### 3. Triggered Updates
#### A. Cloud Functions
```python
# Example Cloud Function trigger
def update_games_features(event, context):
    """Triggered by a change to a Cloud Storage bucket."""
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig()
    
    query = """
    CALL `bgg_data_dev.refresh_games_features_materialized`()
    """
    
    query_job = client.query(query, job_config=job_config)
    query_job.result()  # Wait for the job to complete
```
- Trigger view refresh based on specific events
  - New data ingestion
  - Changes in source tables
  - External system updates

### 4. Incremental Updates
- The materialized view is designed with `PARTITION BY DATE(last_updated)`
- This allows for efficient incremental updates
- Only changed or new data can be processed quickly

## Recommended Update Strategy

1. **Scheduled Daily Refresh**: Set up a daily scheduled query to update the view
2. **Event-Driven Triggers**: Add Cloud Functions for immediate updates on significant changes
3. **Manual Override**: Keep the manual refresh option for urgent updates

## Monitoring and Logging

- The `materialized_view_refresh_log` table tracks:
  - View name
  - Refresh timestamp
  - Status (SUCCESS/FAILED)
  - Error messages (if applicable)

## Best Practices

- Choose update frequency based on:
  - Data change rate
  - Computational cost
  - Freshness requirements
- Monitor refresh logs for any issues
- Adjust update strategy as data patterns evolve
