set dotenv-load

# Defaults — override per-invocation, e.g.
#   just username=alice train outcome=own candidate=lgbm_default
username := "phenrickson"
environment := "dev"
local_root := "models/collections"

# Show available recipes
default:
    @just --list

# Fetch a user's collection from BGG and upsert into BigQuery.
# Run this before `sweep` for a user whose collection has not been
# loaded yet.
load:
    uv run python -m src.collection.load \
        --username {{username}} --environment {{environment}}

# Persist canonical train/val/test splits for an outcome.
split outcome="own":
    uv run python -m src.collection.split \
        --username {{username}} --environment {{environment}} --outcome {{outcome}} \
        --local-root {{local_root}}

# Train one candidate (named in config.collections.candidates) against
# the latest canonical splits.
train outcome="own" candidate="lgbm_default" splits_version="":
    uv run python -m src.collection.train \
        --username {{username}} --environment {{environment}} --outcome {{outcome}} \
        --candidate {{candidate}} \
        --local-root {{local_root}} \
        $([ -n "{{splits_version}}" ] && echo "--splits-version {{splits_version}}")

# Train every candidate listed in config.collections.candidates for an outcome.
# Continue-on-error: runs every candidate, exits non-zero at the end if any failed.
train-all outcome="own":
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
            --username {{username}} --environment {{environment}} --outcome {{outcome}} \
            --candidate "$c" --local-root {{local_root}}; then
            failed+=("$c")
        fi
    done
    if [ ${#failed[@]} -gt 0 ]; then
        echo "FAILED: ${failed[@]}" >&2
        exit 1
    fi

# Print or write a comparison table for an outcome.
compare outcome="own" out="" candidates="":
    uv run python -m src.collection.compare \
        --username {{username}} --environment {{environment}} --outcome {{outcome}} \
        --local-root {{local_root}} \
        $([ -n "{{out}}" ] && echo "--out {{out}}") \
        $([ -n "{{candidates}}" ] && echo "--candidates {{candidates}}")

# Refit a trained candidate on train+val+test through finalize_through.
# Defaults to collections.finalize_through from config.yaml; override with
# finalize_through=2025 if you need a different cutoff.
finalize outcome="own" candidate="lgbm_default" version="latest" finalize_through="":
    uv run python -m src.collection.finalize \
        --username {{username}} --environment {{environment}} --outcome {{outcome}} \
        --candidate {{candidate}} \
        $([ "{{version}}" != "latest" ] && echo "--version {{version}}") \
        $([ -n "{{finalize_through}}" ] && echo "--finalize-through {{finalize_through}}") \
        --local-root {{local_root}}

# Finalize every candidate listed in config.collections.candidates for an outcome.
# Continue-on-error: runs every candidate, exits non-zero at the end if any failed.
finalize-all outcome="own" finalize_through="":
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
            --username {{username}} --environment {{environment}} --outcome {{outcome}} \
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
promote outcome="own" candidate="lgbm_default" version="latest" description="":
    uv run python -m services.collections.register_model \
        --username {{username}} --environment {{environment}} --outcome {{outcome}} \
        --candidate {{candidate}} --version {{version}} \
        --local-root {{local_root}} \
        --description "$([ -n "{{description}}" ] && echo "{{description}}" || echo "{{candidate}} for {{username}}/{{outcome}}")"

# Register one candidate across multiple outcomes in one shot.
#   just promote-many outcomes="own,ever_owned,rated" candidate=lgbm_row_norm
promote-many outcomes candidate="lgbm_default" version="latest" description="":
    uv run python -m services.collections.register_all \
        --username {{username}} --environment {{environment}} \
        --outcomes "{{outcomes}}" \
        --candidate {{candidate}} --version {{version}} \
        --local-root {{local_root}} \
        --description "$([ -n "{{description}}" ] && echo "{{description}}" || echo "{{candidate}} for {{username}}")"

# Promote the configured candidate (collections.deploy.{outcome}.candidate)
# for every user in collections.users. Skips users without a finalized
# artifact for that candidate. Continue-on-error; exits non-zero if any user
# genuinely failed.
#
#   just promote-all
#   just promote-all outcome=own
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
        if just username=$u promote {{outcome}} $cand; then \
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
#   just verify outcome=own
verify outcome="":
    uv run python -m services.collections.verify_models \
        --username {{username}} \
        $([ -n "{{outcome}}" ] && echo "--outcome {{outcome}}")

# End-to-end experiment cycle: split → train all → compare.
# Always runs `compare` if `split` succeeded, even when some candidates fail.
# Exits non-zero if any candidate failed, so cron/CI still notices.
sweep outcome="own":
    #!/usr/bin/env bash
    set -e
    just username={{username}} split {{outcome}}
    set +e
    just username={{username}} train-all {{outcome}}
    train_status=$?
    just username={{username}} compare {{outcome}}
    exit $train_status

# Train all candidates and compare against the most recent existing split.
# Same as `sweep` but skips the split step — use when iterating on candidates
# against a fixed split.
train-compare outcome="own":
    #!/usr/bin/env bash
    set +e
    just username={{username}} train-all {{outcome}}
    train_status=$?
    just username={{username}} compare {{outcome}}
    exit $train_status

# Sweep across a list of users. Skips users who already have at least
# one trained candidate for the outcome. Continue-on-error.
#   just users-sweep "alice bob carol"
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
        if ! just username=$u sweep {{outcome}}; then
            failed+=("$u")
        fi
    done
    if [ ${#failed[@]} -gt 0 ]; then
        echo "FAILED: ${failed[@]}" >&2
        exit 1
    fi
