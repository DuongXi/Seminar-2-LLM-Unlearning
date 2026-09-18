# Train NPO (Negative Preference Optimization) qua pkg_halluc/training/au/train_au.py, xem --help
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_au_common.sh" npo "$@"
