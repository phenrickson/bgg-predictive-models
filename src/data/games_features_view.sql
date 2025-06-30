-- Create a comprehensive view of game features with full names

CREATE OR REPLACE VIEW `bgg_data_dev.games_features` AS
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
        ARRAY_LENGTH(ARRAY_AGG(DISTINCT des.name)) as designer_count
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

-- Optional: Add a comment to describe the view
COMMENT ON VIEW `bgg_data_dev.games_features` 
IS 'Comprehensive view of board game features with full names for categories, mechanics, publishers, designers, artists, and families';
