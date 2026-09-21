# Train GA
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/train_tri_mask_common.sh" ga "$@"
