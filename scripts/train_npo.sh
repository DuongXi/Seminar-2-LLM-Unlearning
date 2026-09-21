# Train NPO (Negative Preference Optimization) qua pkg_halluc/training/tri_mask/train_tri_mask.py, xem --help
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_tri_mask_common.sh" npo "$@"
