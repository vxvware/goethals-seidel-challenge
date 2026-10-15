#!/bin/sh
set -eu
GS_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec uv run --project "$GS_ROOT" python "$GS_ROOT/tools/gs" "$@"
