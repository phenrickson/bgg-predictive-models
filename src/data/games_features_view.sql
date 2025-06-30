-- Create a comprehensive view of game features with full names

CREATE OR REPLACE VIEW `{dataset}.games_features` AS
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
        ARRAY_AGG(DISTINCT fam.name) as families
    FROM `{dataset}.games_active` g
    LEFT JOIN `{dataset}.game_categories` gc
        ON g.game_id = gc.game_id
    LEFT JOIN `{dataset}.categories` cat
        ON gc.category_id = cat.category_id
    LEFT JOIN `{dataset}.game_mechanics` gm
        ON g.game_id = gm.game_id
    LEFT JOIN `{dataset}.mechanics` mech
        ON gm.mechanic_id = mech.mechanic_id
    LEFT JOIN `{dataset}.game_publishers` gp
        ON g.game_id = gp.game_id
    LEFT JOIN `{dataset}.publishers` pub
        ON gp.publisher_id = pub.publisher_id
    LEFT JOIN `{dataset}.game_designers` gd
        ON g.game_id = gd.game_id
    LEFT JOIN `{dataset}.designers` des
        ON gd.designer_id = des.designer_id
    LEFT JOIN `{dataset}.game_artists` ga
        ON g.game_id = ga.game_id
    LEFT JOIN `{dataset}.artists` art
        ON ga.artist_id = art.artist_id
    LEFT JOIN `{dataset}.game_families` gf
        ON g.game_id = gf.game_id
    LEFT JOIN `{dataset}.families` fam
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

-- Note: BigQuery doesn't support COMMENT ON VIEW syntax
-- To add a description to the view, use the Google Cloud Console or the API
