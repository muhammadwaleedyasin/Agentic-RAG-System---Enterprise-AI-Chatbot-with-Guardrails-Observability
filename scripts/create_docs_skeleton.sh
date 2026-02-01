#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="docs"

sections=("getting-started" "guides" "reference" "architecture" "deployment" "security" "monitoring" "examples" "help")

for sec in "${sections[@]}"; do
  for level in {1..5}; do
    dir="$BASE_DIR/$sec/level-$level"
    mkdir -p "$dir"
    idx="$dir/INDEX.md"
    if [ ! -f "$idx" ]; then
      cat > "$idx" <<EOF
# ${sec^} — Level ${level}
This is a placeholder INDEX for ${sec} - Level ${level}.

- Overview: TODO
- Learning objectives: TODO
- 3-click pathways: TODO
- Related reference: TODO

Central hub: ../../INDEX.md
EOF
    fi
  done

  sidx="$BASE_DIR/$sec/INDEX.md"
  if [ ! -f "$sidx" ]; then
    cat > "$sidx" <<EOF
# ${sec^}
Progressive disclosure levels:
- level-1/ (Overview / Beginner)
- level-2/ (Learn-by-doing)
- level-3/ (How-to / Tasks)
- level-4/ (Patterns / Reuse)
- level-5/ (Expert / Design Decisions)

See central hub: ../INDEX.md
EOF
  fi
done

declare -A copies
copies["PRODUCTION_DEPLOYMENT.md"]="deployment/PRODUCTION_DEPLOYMENT.md"
copies["observability-setup.md"]="monitoring/observability-setup.md"
copies["SECURITY_HARDENING.md"]="security/SECURITY_HARDENING.md"
copies["security_implementation_guide.md"]="security/security_implementation_guide.md"
copies["RAG_PIPELINE_USAGE.md"]="guides/RAG_PIPELINE_USAGE.md"
copies["LLM_PROVIDER_GUIDE.md"]="reference/LLM_PROVIDER_GUIDE.md"
copies["LLM_PROVIDER_IMPLEMENTATION.md"]="reference/LLM_PROVIDER_IMPLEMENTATION.md"
copies["FASTAPI_BACKEND_INTEGRATION.md"]="guides/FASTAPI_BACKEND_INTEGRATION.md"
copies["MEMORY_SYSTEM_GUIDE.md"]="guides/MEMORY_SYSTEM_GUIDE.md"
copies["TESTING.md"]="reference/TESTING.md"
copies["SCALING_GUIDE.md"]="deployment/SCALING_GUIDE.md"

for src in "${!copies[@]}"; do
  if [ -f "docs/$src" ] && [ ! -f "docs/${copies[$src]}" ]; then
    mkdir -p "$(dirname "docs/${copies[$src]}")"
    cp "docs/$src" "docs/${copies[$src]}"
  fi
done

if [ -f "docs/accessibility/accessibility-guidelines.md" ] && [ ! -f "docs/getting-started/level-1/accessibility-guidelines.md" ]; then
  mkdir -p "docs/getting-started/level-1"
  cp "docs/accessibility/accessibility-guidelines.md" "docs/getting-started/level-1/accessibility-guidelines.md"
fi

mkdir -p "docs/roles"
for role in frontend backend devops data product; do
  rolefile="docs/roles/${role}.md"
  if [ ! -f "$rolefile" ]; then
    cat > "$rolefile" <<EOF
# ${role^} Role
Placeholder role entry for ${role^}. Add quick tasks, links to central hub, and 3-click pathways.
See: ../INDEX.md
EOF
  fi
done

echo "Documentation skeleton created and copies performed."