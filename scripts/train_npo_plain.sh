# Train NPO-plain
set -euo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_plain_common.sh" npo_plain "$@"
