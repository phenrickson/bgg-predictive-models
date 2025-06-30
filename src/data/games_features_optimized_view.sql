-- Optimized view of game features for predictive models
-- Addresses performance issues and NULL array handling

CREATE OR REPLACE VIEW `{dataset}.games_features` AS
WITH 
-- Pre-aggregate categories to avoid multiple joins
categories_agg AS (
    SELECT
        gc.game_id,
        ARRAY_AGG(cat.name IGNORE NULLS) AS categories
    FROM `{dataset}.game_categories` gc
    LEFT JOIN `{dataset}.categories` cat
        ON gc.category_id = cat.category_id
    GROUP BY gc.game_id
),

-- Pre-aggregate mechanics to avoid multiple joins
mechanics_agg AS (
    SELECT
        gm.game_id,
        ARRAY_AGG(mech.name IGNORE NULLS) AS mechanics
    FROM `{dataset}.game_mechanics` gm
    LEFT JOIN `{dataset}.mechanics` mech
        ON gm.mechanic_id = mech.mechanic_id
    GROUP BY gm.game_id
),

-- Pre-aggregate publishers to avoid multiple joins
publishers_agg AS (
    SELECT
        gp.game_id,
        ARRAY_AGG(pub.name IGNORE NULLS) AS publishers
    FROM `{dataset}.game_publishers` gp
    LEFT JOIN `{dataset}.publishers` pub
        ON gp.publisher_id = pub.publisher_id
    GROUP BY gp.game_id
),

-- Pre-aggregate designers to avoid multiple joins
designers_agg AS (
    SELECT
        gd.game_id,
        ARRAY_AGG(des.name IGNORE NULLS) AS designers
    FROM `{dataset}.game_designers` gd
    LEFT JOIN `{dataset}.designers` des
        ON gd.designer_id = des.designer_id
    GROUP BY gd.game_id
),

-- Pre-aggregate artists to avoid multiple joins
artists_agg AS (
    SELECT
        ga.game_id,
        ARRAY_AGG(art.name IGNORE NULLS) AS artists
    FROM `{dataset}.game_artists` ga
    LEFT JOIN `{dataset}.artists` art
        ON ga.artist_id = art.artist_id
    GROUP BY ga.game_id
),

-- Pre-aggregate families to avoid multiple joins
families_agg AS (
    SELECT
        gf.game_id,
        ARRAY_AGG(fam.name IGNORE NULLS) AS families
    FROM `{dataset}.game_families` gf
    LEFT JOIN `{dataset}.families` fam
        ON gf.family_id = fam.family_id
    GROUP BY gf.game_id
)

-- Main query with efficient joins to pre-aggregated data
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
    -- Use IFNULL to handle missing arrays
    IFNULL(c.categories, []) AS categories,
    IFNULL(m.mechanics, []) AS mechanics,
    IFNULL(p.publishers, []) AS publishers,
    IFNULL(d.designers, []) AS designers,
    IFNULL(a.artists, []) AS artists,
    IFNULL(f.families, []) AS families
FROM `{dataset}.games_active` g
LEFT JOIN categories_agg c ON g.game_id = c.game_id
LEFT JOIN mechanics_agg m ON g.game_id = m.game_id
LEFT JOIN publishers_agg p ON g.game_id = p.game_id
LEFT JOIN designers_agg d ON g.game_id = d.game_id
LEFT JOIN artists_agg a ON g.game_id = a.game_id
LEFT JOIN families_agg f ON g.game_id = f.game_id;

-- Note: This optimized view:
-- 1. Uses CTEs to pre-aggregate related entities, reducing the number of rows processed
-- 2. Uses ARRAY_AGG with IGNORE NULLS to handle NULL values
-- 3. Uses IFNULL to ensure arrays are never NULL
-- 4. Can be easily converted to a materialized view for better performance
