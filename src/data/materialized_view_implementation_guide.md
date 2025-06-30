# Implementing Materialized View Updates for Games Features

## Prerequisites
- Google Cloud Platform account
- BigQuery access
- gcloud CLI or Google Cloud Console access
- (Optional) Python and Cloud Functions development environment

## Implementation Approaches

### 1. Manual Refresh via gcloud CLI

```bash
# Authenticate with Google Cloud
gcloud auth login

# Set the project
gcloud config set project YOUR_PROJECT_ID

# Manually refresh the materialized view
bq query --use_legacy_sql=false \
  'CALL `bgg_data_dev.refresh_games_features_materialized`()'
```

### 2. Scheduled Query Setup (Google Cloud Console)

#### A. Using gcloud CLI
```bash
# Create a scheduled query to refresh daily at midnight
bq mk --transfer_config \
  --project_id=YOUR_PROJECT_ID \
  --data_source=scheduled_query \
  --display_name="Daily Games Features Refresh" \
  --params='{"query":"CALL `bgg_data_dev.refresh_games_features_materialized`()"}' \
  --schedule="every 24 hours"
```

#### B. Manual Console Configuration
1. Go to BigQuery in Google Cloud Console
2. Navigate to Scheduled Queries
3. Create New Scheduled Query
   - Query: `CALL `bgg_data_dev.refresh_games_features_materialized`()`
   - Schedule: Daily at 00:00 UTC
   - Destination: Logging/Monitoring

### 3. Cloud Functions Trigger (Python)

#### 3.1 Create Cloud Function
```python
import functions_framework
from google.cloud import bigquery

@functions_framework.cloud_event
def update_games_features(cloud_event):
    """
    Cloud Function to refresh materialized view
    Triggered by Pub/Sub, Cloud Storage, or other events
    """
    client = bigquery.Client()
    
    # Refresh materialized view
    query = """
    CALL `bgg_data_dev.refresh_games_features_materialized`()
    """
    
    # Execute query
    job_config = bigquery.QueryJobConfig()
    query_job = client.query(query, job_config=job_config)
    
    # Wait for job to complete
    query_job.result()
    
    print("Materialized view refreshed successfully")
```

#### 3.2 Deployment Script
```bash
# Deploy Cloud Function
gcloud functions deploy update_games_features \
  --gen2 \
  --runtime=python310 \
  --source=. \
  --entry-point=update_games_features \
  --trigger-topic=data-update \
  --project=YOUR_PROJECT_ID
```

### 4. Terraform Implementation (Infrastructure as Code)

```hcl
# BigQuery Scheduled Query Resource
resource "google_bigquery_job_schedule" "games_features_refresh" {
  project     = var.project_id
  location    = "US"
  schedule    = "0 0 * * *"  # Daily at midnight
  
  query {
    query = "CALL `bgg_data_dev.refresh_games_features_materialized`()"
    
    destination_table {
      project_id = var.project_id
      dataset_id = "bgg_data_dev"
      table_id   = "games_features_refresh_log"
    }
  }
}
```

## Monitoring and Logging

1. Check Materialized View Refresh Logs
```sql
SELECT * FROM `bgg_data_dev.materialized_view_refresh_log`
ORDER BY refresh_timestamp DESC
LIMIT 10
```

2. Set up Cloud Monitoring Alerts
- Create alerts for failed refreshes
- Monitor query execution time
- Track materialized view size and update frequency

## Best Practices

1. Test refresh mechanisms thoroughly
2. Start with less frequent updates
3. Monitor performance impact
4. Adjust strategy based on data change patterns
5. Implement error handling and notifications

## Recommended Workflow

1. Manual testing of refresh procedure
2. Implement scheduled daily refresh
3. Add event-driven triggers for critical updates
4. Set up comprehensive monitoring
5. Continuously optimize based on performance metrics
