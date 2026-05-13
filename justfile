set dotenv-load

# Defaults — every single-user recipe takes `user` as its first
# positional argument. Pass it like a CLI:
#
#   just collection-train rahdo own lgbm_default
#   just collection-finalize rahdo own lgbm_row_norm
#   just collection-promote rahdo
#
# Bare invocations (`just collection-train`) fall back to the `username` variable
# below. Override per-invocation only if you really need to:
#
#   just username=alice collection-train
username := "phenrickson"
# Pull `environment` from the .env file (loaded via `set dotenv-load`
# above). Falls back to "dev" if .env doesn't define ENVIRONMENT.
environment := env("ENVIRONMENT", "dev")
local_root := "models/collections"

# Default snapshot version for every bgg-* recipe. Bump this when you build
# a new snapshot. Override per-invocation with `just bgg-... snapshot=2`.
snapshot := "1"

# Show available recipes (in file order, not alphabetical).
default:
    @just --list --unsorted

# Show only BGG game-model recipes (snapshot pipeline).
bgg:
    @just --list --unsorted | grep -E '^    bgg' || true

# Show only collection-model recipes.
collection:
    @just --list --unsorted | grep -E '^    collection' || true

# Fetch a user's collection from BGG and upsert into BigQuery.
# Run this before `collection-sweep` for a user whose collection has not been
# loaded yet.
collection-load user=username:
    uv run python -m src.collection.load \
        --username {{user}} --environment {{environment}}

# Persist canonical train/val/test splits for an outcome.
collection-split user=username outcome="own":
    uv run python -m src.collection.split \
        --username {{user}} --environment {{environment}} --outcome {{outcome}} \
        --local-root {{local_root}}

# Train one candidate (named in config.collections.candidates) against
# the latest canonical splits.
collection-train user=username outcome="own" candidate="lgbm_default" splits_version="":
    uv run python -m src.collection.train \
        --username {{user}} --environment {{environment}} --outcome {{outcome}} \
        --candidate {{candidate}} \
        --local-root {{local_root}} \
        $([ -n "{{splits_version}}" ] && echo "--splits-version {{splits_version}}")

# Train every candidate listed in config.collections.candidates for an outcome.
# Continue-on-error: runs every candidate, exits non-zero at the end if any failed.
collection-train-all user=username outcome="own":
    #!/usr/bin/env bash
    failed=()
    candidates=$(uv run python -c 'from src.collection.candidates import load_candidates; from src.utils.config import load_config; print("\n".join(load_candidates(load_config().raw_config)))')
    if [ -z "$candidates" ]; then
        echo "No candidates defined in config.collections.candidates" >&2
        exit 1
    fi
    for c in $candidates; do
        echo "--- $c ---"
        if ! uv run python -m src.collection.train \
            --username {{user}} --environment {{environment}} --outcome {{outcome}} \
            --candidate "$c" --local-root {{local_root}}; then
            failed+=("$c")
        fi
    done
    if [ ${#failed[@]} -gt 0 ]; then
        echo "FAILED: ${failed[@]}" >&2
        exit 1
    fi

# Print or write a comparison table for an outcome.
collection-compare user=username outcome="own" out="" candidates="":
    uv run python -m src.collection.compare \
        --username {{user}} --environment {{environment}} --outcome {{outcome}} \
        --local-root {{local_root}} \
        $([ -n "{{out}}" ] && echo "--out {{out}}") \
        $([ -n "{{candidates}}" ] && echo "--candidates {{candidates}}")

# Refit a trained candidate on train+val+test through finalize_through.
# Defaults to collections.finalize_through from config.yaml; override with
# finalize_through=2025 if you need a different cutoff.
collection-finalize user=username outcome="own" candidate="lgbm_default" version="latest" finalize_through="":
    uv run python -m src.collection.finalize \
        --username {{user}} --environment {{environment}} --outcome {{outcome}} \
        --candidate {{candidate}} \
        $([ "{{version}}" != "latest" ] && echo "--version {{version}}") \
        $([ -n "{{finalize_through}}" ] && echo "--finalize-through {{finalize_through}}") \
        --local-root {{local_root}}

# Finalize every candidate listed in config.collections.candidates for an outcome.
# Continue-on-error: runs every candidate, exits non-zero at the end if any failed.
collection-finalize-all user=username outcome="own" finalize_through="":
    #!/usr/bin/env bash
    failed=()
    candidates=$(uv run python -c 'from src.collection.candidates import load_candidates; from src.utils.config import load_config; print("\n".join(load_candidates(load_config().raw_config)))')
    if [ -z "$candidates" ]; then
        echo "No candidates defined in config.collections.candidates" >&2
        exit 1
    fi
    for c in $candidates; do
        echo "--- $c ---"
        if ! uv run python -m src.collection.finalize \
            --username {{user}} --environment {{environment}} --outcome {{outcome}} \
            --candidate "$c" --local-root {{local_root}} \
            $([ -n "{{finalize_through}}" ] && echo "--finalize-through {{finalize_through}}"); then
            failed+=("$c")
        fi
    done
    if [ ${#failed[@]} -gt 0 ]; then
        echo "FAILED: ${failed[@]}" >&2
        exit 1
    fi

# Register a finalized collection model to GCS for the standalone scoring
# service AND insert a row in the BQ registry. Strict-finalized: requires
# finalized.pkl (run `collection-finalize` first).
collection-promote user=username outcome="own" candidate="lgbm_default" version="latest" description="":
    uv run python -m services.collections.register_model \
        --username {{user}} --environment {{environment}} --outcome {{outcome}} \
        --candidate {{candidate}} --version {{version}} \
        --local-root {{local_root}} \
        --description "$([ -n "{{description}}" ] && echo "{{description}}" || echo "{{candidate}} for {{user}}/{{outcome}}")"

# Register one candidate across multiple outcomes in one shot.
#   just collection-promote-many rahdo "own,ever_owned,rated" lgbm_row_norm
collection-promote-many user=username outcomes="own" candidate="lgbm_default" version="latest" description="":
    uv run python -m services.collections.register_all \
        --username {{user}} --environment {{environment}} \
        --outcomes "{{outcomes}}" \
        --candidate {{candidate}} --version {{version}} \
        --local-root {{local_root}} \
        --description "$([ -n "{{description}}" ] && echo "{{description}}" || echo "{{candidate}} for {{user}}")"

# Promote the configured candidate (collections.deploy.{outcome}.candidate)
# for every user in collections.users. Skips users without a finalized
# artifact for that candidate. Continue-on-error; exits non-zero if any user
# genuinely failed.
#
#   just collection-promote-all
#   just collection-promote-all own
collection-promote-all outcome="own":
    @users=$(uv run python -c "import yaml; \
        c = yaml.safe_load(open('config.yaml')); \
        print('\n'.join(c['collections']['users']))"); \
    cand=$(uv run python -c "import yaml; \
        c = yaml.safe_load(open('config.yaml')); \
        print(c['collections']['deploy']['{{outcome}}']['candidate'])"); \
    deployed=0; skipped=0; failed=0; \
    while IFS= read -r u; do \
        [ -z "$u" ] && continue; \
        path="{{local_root}}/$u/{{outcome}}/$cand"; \
        if ! ls $path/v*/finalized.pkl 2>/dev/null | grep -q .; then \
            echo "skip $u: no finalized.pkl under $path"; \
            skipped=$((skipped + 1)); \
            continue; \
        fi; \
        echo "=== collection-promote $u {{outcome}} $cand ==="; \
        if just collection-promote $u {{outcome}} $cand; then \
            deployed=$((deployed + 1)); \
        else \
            echo "FAIL: $u"; \
            failed=$((failed + 1)); \
        fi; \
    done <<< "$users"; \
    echo "collection-promote-all: deployed=$deployed skipped=$skipped failed=$failed"; \
    [ $failed -eq 0 ]

# List registered collection models for a user from GCS.
#   just collection-verify
#   just collection-verify rahdo
#   just collection-verify rahdo own
collection-verify user=username outcome="":
    uv run python -m services.collections.verify_models \
        --username {{user}} \
        $([ -n "{{outcome}}" ] && echo "--outcome {{outcome}}")

# End-to-end experiment cycle: collection-split → collection-train-all → collection-compare.
# Always runs `collection-compare` if `collection-split` succeeded, even when some candidates fail.
# Exits non-zero if any candidate failed, so cron/CI still notices.
collection-sweep user=username outcome="own":
    #!/usr/bin/env bash
    set -e
    just collection-split {{user}} {{outcome}}
    set +e
    just collection-train-all {{user}} {{outcome}}
    train_status=$?
    just collection-compare {{user}} {{outcome}}
    exit $train_status

# Train all candidates and compare against the most recent existing split.
# Same as `collection-sweep` but skips the split step — use when iterating on
# candidates against a fixed split.
collection-train-compare user=username outcome="own":
    #!/usr/bin/env bash
    set +e
    just collection-train-all {{user}} {{outcome}}
    train_status=$?
    just collection-compare {{user}} {{outcome}}
    exit $train_status

# Sweep across a list of users. Skips users who already have at least
# one trained candidate for the outcome. Continue-on-error.
#   just collection-users-sweep "alice bob carol"
#   just collection-users-sweep "alice bob" ever_owned
collection-users-sweep users outcome="own":
    #!/usr/bin/env bash
    shopt -s nullglob
    failed=()
    for u in {{users}}; do
        candidate_dirs=({{local_root}}/{{environment}}/$u/{{outcome}}/*/v*)
        if [ ${#candidate_dirs[@]} -gt 0 ]; then
            echo "skip $u (already processed)"
            continue
        fi
        echo "===== $u ====="
        if ! just collection-sweep $u {{outcome}}; then
            failed+=("$u")
        fi
    done
    if [ ${#failed[@]} -gt 0 ]; then
        echo "FAILED: ${failed[@]}" >&2
        exit 1
    fi

# Render the collection report for a user. Reads artifacts from
# models/collections/ locally and pulls collection/games/upcoming
# predictions from BigQuery (set BGG_REPORTS_OFFLINE=1 in .env to
# stub the BQ-backed sections).
#   just collection-render                          # default user
#   just collection-render GOBBluth89               # other user
#   just collection-render GOBBluth89 ever_owned    # other user + outcome
collection-render user=username outcome="own" candidate="":
    uv run python -m reports.render \
        --username {{user}} --outcome {{outcome}} \
        $([ -n "{{candidate}}" ] && echo "--candidate {{candidate}}")

# Render the collection report for every user under models/collections/.
collection-render-all outcome="own":
    uv run python -m reports.render --all-users --outcome {{outcome}}

# Render the index page (reports/index.qmd) which lists all users with
# finalized models and links to their per-user reports. Builds from the
# local artifact tree by default; pass `source=gs://...` to build from
# cloud storage instead (used by the CI workflow).
collection-render-index source="local":
    uv run python -m reports.build_index --source {{source}}

# Render the report against synthetic fixture data — no BQ, no artifacts.
# Use this for fast iteration on styling/layout: edits to the qmd, css,
# or viz code can be checked in seconds rather than waiting on real loads.
collection-render-sandbox:
    uv run python -m reports.render --fixture

# --- Artifact sync to GCS ---
#
# Local experiment artifacts live under models/collections/<user>/...
# Mirror them into gs://<bucket>/<env>/collections/<user>/... so the
# scheduled reports workflow (CI) can render directly from cloud
# storage. Only finalized versions are useful for the report; we sync
# the whole tree so future runs pick up new candidates automatically.
gcs_artifacts_root := "gs://bgg-predictive-models/" + environment + "/collections"

# Sync one user's collection artifacts to GCS.
#   just collection-sync-artifacts rahdo
#   just collection-sync-artifacts rahdo --prune    # remove cloud files not in local
collection-sync-artifacts user=username prune="":
    #!/usr/bin/env bash
    set -e
    src="{{local_root}}/{{user}}"
    dst="{{gcs_artifacts_root}}/{{user}}"
    if [ ! -d "$src" ]; then
        echo "No local artifacts for user '{{user}}' at $src" >&2
        exit 1
    fi
    echo "syncing $src -> $dst"
    if [ -n "{{prune}}" ]; then
        gsutil -m rsync -r -d "$src" "$dst"
    else
        gsutil -m rsync -r "$src" "$dst"
    fi

# Sync every local user's collection artifacts to GCS. Skips users with
# no local directory; continue-on-error.
collection-sync-artifacts-all prune="":
    #!/usr/bin/env bash
    failed=()
    for user_dir in {{local_root}}/*/; do
        u=$(basename "$user_dir")
        echo "===== $u ====="
        if ! just collection-sync-artifacts "$u" {{prune}}; then
            failed+=("$u")
        fi
    done
    if [ ${#failed[@]} -gt 0 ]; then
        echo "FAILED: ${failed[@]}" >&2
        exit 1
    fi

# Pull one user's collection artifacts FROM GCS into the local tree.
# Use this on a fresh machine to seed models/collections/ from the
# canonical cloud copy. No prune by default — pass `--prune` to also
# remove local files missing from the cloud.
#   just collection-pull-artifacts rahdo
#   just collection-pull-artifacts rahdo --prune
collection-pull-artifacts user=username prune="":
    #!/usr/bin/env bash
    set -e
    src="{{gcs_artifacts_root}}/{{user}}"
    dst="{{local_root}}/{{user}}"
    mkdir -p "$dst"
    echo "pulling $src -> $dst"
    if [ -n "{{prune}}" ]; then
        gsutil -m rsync -r -d "$src" "$dst"
    else
        gsutil -m rsync -r "$src" "$dst"
    fi

# Pull every user under gs://.../collections/ into the local tree.
# Continue-on-error.
collection-pull-artifacts-all prune="":
    #!/usr/bin/env bash
    failed=()
    users=$(gsutil ls "{{gcs_artifacts_root}}/" 2>/dev/null \
        | sed -e 's|/$||' -e "s|^{{gcs_artifacts_root}}/||")
    if [ -z "$users" ]; then
        echo "No users found at {{gcs_artifacts_root}}/" >&2
        exit 1
    fi
    for u in $users; do
        echo "===== $u ====="
        if ! just collection-pull-artifacts "$u" {{prune}}; then
            failed+=("$u")
        fi
    done
    if [ ${#failed[@]} -gt 0 ]; then
        echo "FAILED: ${failed[@]}" >&2
        exit 1
    fi

# --- BGG game models (snapshot pipeline) ---
#
# Pipeline order: build snapshot → build splits → train cascade →
# finalize on train+tune+test → simulate eval_year. Recipes are
# defined below in that order; `just --list --unsorted` shows them
# that way.
#
# Most-common entry points:
#   just bgg-build                     one-time, pulls features from BigQuery
#   just bgg-run-yoy 2021 2022         end-to-end YoY pipeline
#
# Universe-wide models: hurdle, complexity, rating, users_rated, geek_rating.
# Collection-models recipes are prefixed `collection-`.

# === 0. End-to-end ===

# Examples:
#   just bgg-run-yoy 2021 2022
#   just bgg-run-yoy 2022 2022    # single year
#
# End-to-end YoY pipeline for [start, end]: split → train → finalize → simulate.
bgg-run-yoy start end:
    #!/usr/bin/env bash
    set -e
    just bgg-split-yoy {{start}} {{end}}
    just bgg-train-yoy {{start}} {{end}}
    just bgg-finalize-yoy {{start}} {{end}}
    just bgg-simulate-yoy {{start}} {{end}}

# === 1. Build snapshot ===

# Build a versioned data snapshot from BigQuery.
bgg-build:
    uv run python -m src.models.build_snapshot --use-embeddings

# === 2. Build splits ===

# Examples:
#   just bgg-split-yoy 2021 2022
#
# Build the YoY family of splits for years [start, end].
bgg-split-yoy start="2018" end="2024":
    uv run python -m src.models.build_split \
        --snapshot-version {{snapshot}} --yoy --yoy-start {{start}} --yoy-end {{end}}

# Build a single named split from the snapshot.
bgg-split split="standard":
    uv run python -m src.models.build_split \
        --snapshot-version {{snapshot}} --split-name {{split}}

# === 3. Train cascade ===

# Examples:
#   just bgg-train-yoy 2021 2022
#
# Train the cascade across a YoY range. Expands [start, end] into yoy_{start},...,yoy_{end} and hands off to bgg-train-all.
bgg-train-yoy start end:
    #!/usr/bin/env bash
    set -e
    splits=$(seq {{start}} {{end}} | sed 's/^/yoy_/' | paste -sd, -)
    just bgg-train-all "$splits"

# Examples:
#   just bgg-train-all
#   just bgg-train-all standard,yoy_2024
#   just bgg-train-all all
#
# Train the full cascade (hurdle → complexity → rating + users_rated → geek_rating) on one or more splits. `splits=all` resolves to every split in the snapshot.
bgg-train-all splits="standard":
    #!/usr/bin/env bash
    set -e

    cand_for() {
        uv run python -c "from src.models.candidate_config import list_candidates; print(list_candidates('$1')[0])"
    }

    splits="{{splits}}"
    if [ "$splits" = "all" ]; then
        splits=$(uv run python -c "from src.models.snapshot_storage import SnapshotStorage; print(','.join(SnapshotStorage({{snapshot}}).list_splits()))")
        if [ -z "$splits" ]; then
            echo "No splits under snapshot v{{snapshot}}. Run bgg-split-yoy or bgg-split first." >&2
            exit 1
        fi
        echo "Resolved splits=all → $splits"
    fi

    HURDLE=$(cand_for hurdle)
    COMPLEXITY=$(cand_for complexity)
    RATING=$(cand_for rating)
    USERS_RATED=$(cand_for users_rated)
    GEEK_RATING=$(cand_for geek_rating)

    echo "===== hurdle ====="
    uv run python -m src.pipeline.train --model hurdle --candidate "$HURDLE" \
        --snapshot-version {{snapshot}} --splits "$splits"

    echo "===== complexity (train + score) ====="
    uv run python -m src.pipeline.train --model complexity --candidate "$COMPLEXITY" \
        --snapshot-version {{snapshot}} --splits "$splits"
    uv run python -m src.pipeline.score --model complexity --candidate "$COMPLEXITY" \
        --snapshot-version {{snapshot}} --splits "$splits"

    echo "===== users_rated (train + score) ====="
    uv run python -m src.pipeline.train --model users_rated --candidate "$USERS_RATED" \
        --snapshot-version {{snapshot}} --splits "$splits"
    uv run python -m src.pipeline.score --model users_rated --candidate "$USERS_RATED" \
        --snapshot-version {{snapshot}} --splits "$splits"

    echo "===== rating (train + score) ====="
    uv run python -m src.pipeline.train --model rating --candidate "$RATING" \
        --snapshot-version {{snapshot}} --splits "$splits"
    uv run python -m src.pipeline.score --model rating --candidate "$RATING" \
        --snapshot-version {{snapshot}} --splits "$splits"

    echo "===== geek_rating ====="
    uv run python -m src.pipeline.train --model geek_rating --candidate "$GEEK_RATING" \
        --snapshot-version {{snapshot}} --splits "$splits"

    echo "===== done ====="

# Train one candidate against named splits.
bgg-train model="hurdle" candidate="" splits="standard" upstream="":
    #!/usr/bin/env bash
    set -e
    cand="{{candidate}}"
    if [ -z "$cand" ]; then
        cand=$(uv run python -c 'from src.models.candidate_config import list_candidates; print(list_candidates("{{model}}")[0])')
    fi
    uv run python -m src.pipeline.train \
        --model {{model}} --candidate "$cand" \
        --snapshot-version {{snapshot}} --splits {{splits}} \
        $([ -n "{{upstream}}" ] && echo "--upstream {{upstream}}")

# Score one candidate (writes score.parquet under each split).
bgg-score model="complexity" candidate="" splits="standard" upstream="":
    #!/usr/bin/env bash
    set -e
    cand="{{candidate}}"
    if [ -z "$cand" ]; then
        cand=$(uv run python -c 'from src.models.candidate_config import list_candidates; print(list_candidates("{{model}}")[0])')
    fi
    uv run python -m src.pipeline.score \
        --model {{model}} --candidate "$cand" \
        --snapshot-version {{snapshot}} --splits {{splits}} \
        $([ -n "{{upstream}}" ] && echo "--upstream {{upstream}}")

# === 4. Finalize cascade ===

# Examples:
#   just bgg-finalize-yoy 2021 2022
#   just bgg-finalize-yoy 2022 2022    # single year
#
# Finalize the cascade for each YoY split: refit on train+tune+test through split.test_through.
bgg-finalize-yoy start end:
    #!/usr/bin/env bash
    set -e
    cand_for() {
        uv run python -c "from src.models.candidate_config import list_candidates; print(list_candidates('$1')[0])"
    }
    HURDLE=$(cand_for hurdle)
    COMPLEXITY=$(cand_for complexity)
    RATING=$(cand_for rating)
    USERS_RATED=$(cand_for users_rated)
    GEEK_RATING=$(cand_for geek_rating)

    for y in $(seq {{start}} {{end}}); do
        split="yoy_$y"
        echo "===== $split ====="
        for pair in "hurdle=$HURDLE" "complexity=$COMPLEXITY" \
                    "users_rated=$USERS_RATED" "rating=$RATING" \
                    "geek_rating=$GEEK_RATING"; do
            model="${pair%%=*}"
            cand="${pair##*=}"
            echo "--- $model / $cand ($split) ---"
            uv run python -m src.pipeline.finalize \
                --model "$model" --candidate "$cand" \
                --snapshot-version {{snapshot}} --split "$split"
        done
    done

# === 5. Simulate ===

# Examples:
#   just bgg-simulate-yoy 2021 2022
#   just bgg-simulate-yoy 2021 2022 default 200
#
# Simulate across a YoY range. Expands [start, end] into yoy_{start},...,yoy_{end} and hands off to bgg-simulate-all.
bgg-simulate-yoy start end name="" samples="500":
    #!/usr/bin/env bash
    set -e
    splits=$(seq {{start}} {{end}} | sed 's/^/yoy_/' | paste -sd, -)
    just bgg-simulate-all "$splits" "{{name}}" {{samples}}

# Examples:
#   just bgg-simulate-all
#   just bgg-simulate-all yoy_2021,yoy_2022
#   just bgg-simulate-all all default 200
#
# Simulate across one or more splits. `splits=all` resolves to every split in the snapshot.
bgg-simulate-all splits="all" name="" samples="500":
    #!/usr/bin/env bash
    set -e
    splits="{{splits}}"
    if [ "$splits" = "all" ]; then
        splits=$(uv run python -c "from src.models.snapshot_storage import SnapshotStorage; print(' '.join(SnapshotStorage({{snapshot}}).list_splits()))")
        if [ -z "$splits" ]; then
            echo "No splits under snapshot v{{snapshot}}." >&2
            exit 1
        fi
    else
        splits=$(echo "$splits" | tr ',' ' ')
    fi
    for split in $splits; do
        echo "===== $split ====="
        uv run python -m src.pipeline.evaluate_simulation \
            --snapshot-version {{snapshot}} --split $split \
            --n-samples {{samples}} \
            $([ -n "{{name}}" ] && echo "--simulation-name {{name}}")
    done

# Examples:
#   just bgg-simulate
#   just bgg-simulate yoy_2019 default 500
#
# Simulate one split on eval_year = split.test_through + 1. Requires per-split finalize first.
bgg-simulate split="standard" name="" samples="500":
    uv run python -m src.pipeline.evaluate_simulation \
        --snapshot-version {{snapshot}} --split {{split}} \
        --n-samples {{samples}} \
        $([ -n "{{name}}" ] && echo "--simulation-name {{name}}")

# === 6. Plot ===

# Examples:
#   just bgg-plot
#   just bgg-plot yoy_2021 default 50
#
# Re-render top-N geek_rating plot for an existing simulation.
bgg-plot split="standard" name="" top="100" version="":
    #!/usr/bin/env bash
    set -e
    uv run python -m src.pipeline.plot_simulation \
        --snapshot-version {{snapshot}} --split {{split}} \
        --top-n {{top}} \
        $([ -n "{{name}}" ] && echo "--simulation-name {{name}}") \
        $([ -n "{{version}}" ] && echo "--simulation-version {{version}}")
