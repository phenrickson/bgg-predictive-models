-- Materialized View Creation and Update Strategies for Games Features

-- Option 1: Materialized View with Incremental Updates
CREATE OR REPLACE MATERIALIZED VIEW `bgg_data_dev.games_features_materialized`
PARTITION BY DATE(last_updated)
CLUSTER BY game_id
AS
WITH games_features AS (
    SELECT
        g.game_id,
        g.year_published,
        g.average_rating,
        g.average_weight,
        g.users_rated,
        g.min_players,
        g.max_players,
        g.min_playtime,
        g.max_playtime,
        g.min_age,
        g.image,
        g.thumbnail,
        g.description,
        -- Categories with names
        ARRAY_AGG(DISTINCT cat.name) as categories,
        -- Mechanics with names
        ARRAY_AGG(DISTINCT mech.name) as mechanics,
        -- Publishers with names
        ARRAY_AGG(DISTINCT pub.name) as publishers,
        -- Designers with names
        ARRAY_AGG(DISTINCT des.name) as designers,
        -- Artists with names
        ARRAY_AGG(DISTINCT art.name) as artists,
        -- Families with names
        ARRAY_AGG(DISTINCT fam.name) as families,
        -- Counts for additional feature engineering
        ARRAY_LENGTH(ARRAY_AGG(DISTINCT pub.name)) as publisher_count,
        ARRAY_LENGTH(ARRAY_AGG(DISTINCT des.name)) as designer_count,
        -- Tracking last update
        CURRENT_TIMESTAMP() as last_updated
    FROM `bgg_data_dev.games_active` g
    LEFT JOIN `bgg_data_dev.game_categories` gc
        ON g.game_id = gc.game_id
    LEFT JOIN `bgg_data_dev.categories` cat
        ON gc.category_id = cat.category_id
    LEFT JOIN `bgg_data_dev.game_mechanics` gm
        ON g.game_id = gm.game_id
    LEFT JOIN `bgg_data_dev.mechanics` mech
        ON gm.mechanic_id = mech.mechanic_id
    LEFT JOIN `bgg_data_dev.game_publishers` gp
        ON g.game_id = gp.game_id
    LEFT JOIN `bgg_data_dev.publishers` pub
        ON gp.publisher_id = pub.publisher_id
    LEFT JOIN `bgg_data_dev.game_designers` gd
        ON g.game_id = gd.game_id
    LEFT JOIN `bgg_data_dev.designers` des
        ON gd.designer_id = des.designer_id
    LEFT JOIN `bgg_data_dev.game_artists` ga
        ON g.game_id = ga.game_id
    LEFT JOIN `bgg_data_dev.artists` art
        ON ga.artist_id = art.artist_id
    LEFT JOIN `bgg_data_dev.game_families` gf
        ON g.game_id = gf.game_id
    LEFT JOIN `bgg_data_dev.families` fam
        ON gf.family_id = fam.family_id
    GROUP BY 
        g.game_id,
        g.year_published,
        g.average_rating,
        g.average_weight,
        g.users_rated,
        g.min_players,
        g.max_players,
        g.min_playtime,
        g.max_playtime,
        g.min_age,
        g.image,
        g.description,
        g.thumbnail
)
SELECT * FROM games_features;

-- Optional: Create a stored procedure for manual refresh
CREATE OR REPLACE PROCEDURE `bgg_data_dev.refresh_games_features_materialized`()
BEGIN
    REFRESH MATERIALIZED VIEW `bgg_data_dev.games_features_materialized`;
END;

-- Example Scheduled Query (to be configured in Cloud Console or via gcloud)
-- This is a placeholder and needs to be set up in the Google Cloud Console
-- Refresh the materialized view daily at midnight
-- bq mk --transfer_config \
--   --project_id=your-project \
--   --data_source=scheduled_query \
--   --display_name="Daily Games Features Refresh" \
--   --params='{"query":"CALL `bgg_data_dev.refresh_games_features_materialized`()"}' \
--   --schedule="every 24 hours"

-- Monitoring and Logging
-- Create a table to track materialized view refresh history
CREATE TABLE IF NOT EXISTS `bgg_data_dev.materialized_view_refresh_log` (
    view_name STRING,
    refresh_timestamp TIMESTAMP,
    status STRING,
    error_message STRING
);

-- Stored procedure with logging
CREATE OR REPLACE PROCEDURE `bgg_data_dev.log_materialized_view_refresh`()
BEGIN
    BEGIN
        REFRESH MATERIALIZED VIEW `bgg_data_dev.games_features_materialized`;
        
        INSERT INTO `bgg_data_dev.materialized_view_refresh_log` 
        (view_name, refresh_timestamp, status)
        VALUES ('games_features_materialized', CURRENT_TIMESTAMP(), 'SUCCESS');
    EXCEPTION WHEN ERROR THEN
        INSERT INTO `bgg_data_dev.materialized_view_refresh_log` 
        (view_name, refresh_timestamp, status, error_message)
        VALUES ('games_features_materialized', CURRENT_TIMESTAMP(), 'FAILED', @@error.message);
        
        -- Optionally re-raise the error
        RAISE;
    END;
END;

-- Annotations and Documentation
COMMENT ON MATERIALIZED VIEW `bgg_data_dev.games_features_materialized`
IS 'Comprehensive materialized view of board game features with full names, updated periodically. Includes categories, mechanics, publishers, designers, artists, and families.';
