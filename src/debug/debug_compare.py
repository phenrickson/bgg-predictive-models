"""Compare the two preprocessor creation paths."""

from src.features.preprocessor import create_bgg_preprocessor
from src.models.training import create_preprocessing_pipeline

print("=== create_bgg_preprocessor directly ===")
p1 = create_bgg_preprocessor(
    model_type="linear",
    preserve_columns=["year_published"],
    include_description_embeddings=False,
)
print(f"Steps: {list(p1.named_steps.keys())}")
bgg1 = p1.named_steps['bgg_preprocessor']
print(f"BGG params: {bgg1.get_params()}")

print("\n=== create_preprocessing_pipeline ===")
p2 = create_preprocessing_pipeline(
    model_type="linear",
    preserve_columns=["year_published"],
    include_description_embeddings=False,
)
print(f"Steps: {list(p2.named_steps.keys())}")
bgg2 = p2.named_steps['bgg_preprocessor']
print(f"BGG params: {bgg2.get_params()}")

print("\n=== Differences ===")
params1 = bgg1.get_params()
params2 = bgg2.get_params()
for key in set(params1.keys()) | set(params2.keys()):
    v1 = params1.get(key)
    v2 = params2.get(key)
    if v1 != v2:
        print(f"  {key}: {v1} vs {v2}")
