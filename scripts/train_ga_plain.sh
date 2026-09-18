# Train GA-plain
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_plain_common.sh" ga_plain "$@"
