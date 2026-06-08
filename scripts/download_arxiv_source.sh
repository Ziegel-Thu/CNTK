#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <arxiv-id> <paper-dir>" >&2
  echo "Example: $0 2006.13198 paper/001-canatar-2020-spectral-bias" >&2
  exit 2
fi

arxiv_id="$1"
paper_dir="$2"
source_dir="${paper_dir}/source"

mkdir -p "$source_dir"

curl -L "https://arxiv.org/pdf/${arxiv_id}" -o "${paper_dir}/paper.pdf"
curl -L "https://arxiv.org/e-print/${arxiv_id}" -o "${paper_dir}/source.tar"

if tar -tf "${paper_dir}/source.tar" >/dev/null 2>&1; then
  tar -xf "${paper_dir}/source.tar" -C "$source_dir"
else
  cp "${paper_dir}/source.tar" "${source_dir}/main.tex"
fi

cat > "${paper_dir}/README.md" <<EOF
# arXiv:${arxiv_id}

## Files

- \`paper.pdf\`
- \`source.tar\`
- \`source/\`

## Notes

Add project-specific notes here.
EOF
