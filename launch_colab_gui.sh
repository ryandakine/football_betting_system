#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

install_launcher() {
    local desktop_dir="${HOME}/.local/share/applications"
    local desktop_file="${desktop_dir}/colab_gpu_fun_station.desktop"

    mkdir -p "${desktop_dir}"

    cat >"${desktop_file}" <<EOF
[Desktop Entry]
Type=Application
Name=Colab GPU Fun Station
Comment=Open the kid-friendly Colab GPU console
Exec=/usr/bin/env bash -lc 'cd "${SCRIPT_DIR}" && ./launch_colab_gui.sh'
Terminal=false
Categories=Utility;Education;Development;
EOF

    chmod +x "${desktop_file}"
    echo "Launcher installed at ${desktop_file}"
}

if [[ "${1:-}" == "--install-launcher" ]]; then
    install_launcher
    exit 0
fi

if command -v python3 >/dev/null 2>&1; then
    exec python3 colab_gpu_gui.py "$@"
fi

echo "python3 interpreter not found." >&2
exit 1
