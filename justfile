set dotenv-load

# Defaults — every single-user recipe takes `user` as its first
# positional argument. Pass it like a CLI:
#
#   just train rahdo own lgbm_default
#   just finalize rahdo own lgbm_row_norm
#   just promote rahdo
#
# Bare invocations (`just train`) fall back to the `username` variable
# below. Override per-invocation only if you really need to:
#
#   just username=alice train
username := "phenrickson"
environment := "dev"
local_root := "models/collections"

# Show available recipes
default:
    @just --list

# Fetch a user's collection from BGG and upsert into BigQuery.
# Run this before `sweep` for a user whose collection has not been
# loaded yet.
load user=username:
    uv run python -m src.collection.load \
        --username {{user}} --environment {{environment}}

# Persist canonical train/val/test splits for an outcome.
split user=username outcome="own":
    uv run python -m src.collection.split \
        --username {{user}} --environment {{environment}} --outcome {{outcome}} \
        --local-root {{local_root}}

# Train one candidate (named in config.collections.candidates) against
# the latest canonical splits.
train user=username outcome="own" candidate="lgbm_default" splits_version="":
    uv run python -m src.collection.train \
        --username {{user}} --environment {{environment}} --outcome {{outcome}} \
        --candidate {{candidate}} \
        --local-root {{local_root}} \
        $([ -n "{{splits_version}}" ] && echo "--splits-version {{splits_version}}")

# Train every candidate listed in config.collections.candidates for an outcome.
# Continue-on-error: runs every candidate, exits non-zero at the end if any failed.
train-all user=username outcome="own":
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
compare user=username outcome="own" out="" candidates="":
    uv run python -m src.collection.compare \
        --username {{user}} --environment {{environment}} --outcome {{outcome}} \
        --local-root {{local_root}} \
        $([ -n "{{out}}" ] && echo "--out {{out}}") \
        $([ -n "{{candidates}}" ] && echo "--candidates {{candidates}}")

# Refit a trained candidate on train+val+test through finalize_through.
# Defaults to collections.finalize_through from config.yaml; override with
# finalize_through=2025 if you need a different cutoff.
finalize user=username outcome="own" candidate="lgbm_default" version="latest" finalize_through="":
    uv run python -m src.collection.finalize \
        --username {{user}} --environment {{environment}} --outcome {{outcome}} \
        --candidate {{candidate}} \
        $([ "{{version}}" != "latest" ] && echo "--version {{version}}") \
        $([ -n "{{finalize_through}}" ] && echo "--finalize-through {{finalize_through}}") \
        --local-root {{local_root}}

# Finalize every candidate listed in config.collections.candidates for an outcome.
# Continue-on-error: runs every candidate, exits non-zero at the end if any failed.
finalize-all user=username outcome="own" finalize_through="":
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
# finalized.pkl (run `finalize` first).
promote user=username outcome="own" candidate="lgbm_default" version="latest" description="":
    uv run python -m services.collections.register_model \
        --username {{user}} --environment {{environment}} --outcome {{outcome}} \
        --candidate {{candidate}} --version {{version}} \
        --local-root {{local_root}} \
        --description "$([ -n "{{description}}" ] && echo "{{description}}" || echo "{{candidate}} for {{user}}/{{outcome}}")"

# Register one candidate across multiple outcomes in one shot.
#   just promote-many rahdo "own,ever_owned,rated" lgbm_row_norm
promote-many user=username outcomes="own" candidate="lgbm_default" version="latest" description="":
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
#   just promote-all
#   just promote-all own
promote-all outcome="own":
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
        echo "=== promote $u {{outcome}} $cand ==="; \
        if just promote $u {{outcome}} $cand; then \
            deployed=$((deployed + 1)); \
        else \
            echo "FAIL: $u"; \
            failed=$((failed + 1)); \
        fi; \
    done <<< "$users"; \
    echo "promote-all: deployed=$deployed skipped=$skipped failed=$failed"; \
    [ $failed -eq 0 ]

# List registered collection models for a user from GCS.
#   just verify
#   just verify rahdo
#   just verify rahdo own
verify user=username outcome="":
    uv run python -m services.collections.verify_models \
        --username {{user}} \
        $([ -n "{{outcome}}" ] && echo "--outcome {{outcome}}")

# End-to-end experiment cycle: split → train all → compare.
# Always runs `compare` if `split` succeeded, even when some candidates fail.
# Exits non-zero if any candidate failed, so cron/CI still notices.
sweep user=username outcome="own":
    #!/usr/bin/env bash
    set -e
    just split {{user}} {{outcome}}
    set +e
    just train-all {{user}} {{outcome}}
    train_status=$?
    just compare {{user}} {{outcome}}
    exit $train_status

# Train all candidates and compare against the most recent existing split.
# Same as `sweep` but skips the split step — use when iterating on candidates
# against a fixed split.
train-compare user=username outcome="own":
    #!/usr/bin/env bash
    set +e
    just train-all {{user}} {{outcome}}
    train_status=$?
    just compare {{user}} {{outcome}}
    exit $train_status

# Sweep across a list of users. Skips users who already have at least
# one trained candidate for the outcome. Continue-on-error.
#   just users-sweep "alice bob carol"
#   just users-sweep "alice bob" ever_owned
users-sweep users outcome="own":
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
        if ! just sweep $u {{outcome}}; then
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
#   just render                          # default user
#   just render GOBBluth89               # other user
#   just render GOBBluth89 ever_owned    # other user + outcome
render user=username outcome="own" candidate="":
    uv run python -m reports.render \
        --username {{user}} --outcome {{outcome}} \
        $([ -n "{{candidate}}" ] && echo "--candidate {{candidate}}")

# Render the collection report for every user under models/collections/.
render-all outcome="own":
    uv run python -m reports.render --all-users --outcome {{outcome}}

# Render the index page (reports/index.qmd) which lists all users with
# finalized models and links to their per-user reports. Builds from the
# local artifact tree by default; pass `source=gs://...` to build from
# cloud storage instead (used by the CI workflow).
render-index source="local":
    uv run python -m reports.build_index --source {{source}}

# Render the report against synthetic fixture data — no BQ, no artifacts.
# Use this for fast iteration on styling/layout: edits to the qmd, css,
# or viz code can be checked in seconds rather than waiting on real loads.
render-sandbox:
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
#   just sync-artifacts rahdo
#   just sync-artifacts rahdo --prune    # remove cloud files not in local
sync-artifacts user=username prune="":
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
sync-artifacts-all prune="":
    #!/usr/bin/env bash
    failed=()
    for user_dir in {{local_root}}/*/; do
        u=$(basename "$user_dir")
        echo "===== $u ====="
        if ! just sync-artifacts "$u" {{prune}}; then
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
#   just pull-artifacts rahdo
#   just pull-artifacts rahdo --prune
pull-artifacts user=username prune="":
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
pull-artifacts-all prune="":
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
        if ! just pull-artifacts "$u" {{prune}}; then
            failed+=("$u")
        fi
    done
    if [ ${#failed[@]} -gt 0 ]; then
        echo "FAILED: ${failed[@]}" >&2
        exit 1
    fi
