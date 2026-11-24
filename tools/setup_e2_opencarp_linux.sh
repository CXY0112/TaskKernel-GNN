#!/usr/bin/env bash
# Install the pinned official solver after WSL2/Ubuntu is ready. No simulations or training.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
package="$script_dir/downloads/opencarp-v20.0.deb"
expected_sha="1d3572ce8b7c5c2f7e61505247118a507c50a45cece18195b40018eecd14834a"
if [[ ! -f "$package" ]]; then
    echo "Missing official package: $package" >&2
    exit 1
fi
printf '%s  %s\n' "$expected_sha" "$package" | sha256sum --check --strict
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv python3-testresources git python-is-python3 "$package"
runtime_dir="/opt/taskkernel-e2"
sudo mkdir -p "$runtime_dir"
cd "$runtime_dir"
if [[ ! -f venv_carputils/bin/activate ]]; then
    install_carputils.sh
fi
source venv_carputils/bin/activate
export PATH="/usr/local/lib/opencarp/lib/petsc/bin:$PATH"
python -c 'import numpy, carputils; print("carputils:", carputils.__file__)'
openCARP -buildinfo | python -c 'import re,sys; print(re.sub(r"(https?://)[^/\s]*@", r"\1[redacted]@", sys.stdin.read()), end="")'
printf '\nSolver environment installed. Reference simulation and model training have NOT started.\n'
printf 'Next: activate %s/venv_carputils/bin/activate and run a single reference case.\n' "$runtime_dir"
