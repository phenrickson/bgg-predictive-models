"""Investigate HTML entities in game descriptions."""

import re
from collections import Counter

from src.models.training import load_data
from src.utils.config import load_config

config = load_config()
df = load_data(end_train_year=config.years.train_end)

descriptions = df["description"].fill_null("").to_list()

# Look for HTML entities
entity_pattern = re.compile(r"&([a-z]+);")
entity_counts = Counter()

for desc in descriptions:
    entities = entity_pattern.findall(desc)
    entity_counts.update(entities)

print("Top 30 HTML entities found in descriptions:")
print("-" * 40)
for entity, count in entity_counts.most_common(30):
    print(f"&{entity};  -> {count:,}")

# Show some example descriptions with these entities
print("\n\nExample descriptions with &aacute;:")
print("-" * 40)
count = 0
for desc in descriptions:
    if "&aacute;" in desc:
        print(desc[:300])
        print("...")
        print()
        count += 1
        if count >= 3:
            break
