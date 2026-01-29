#!/bin/bash
# Download DIMACS road network datasets for benchmarking
#
# Usage:
#   ./scripts/download-dimacs.sh          # Download default (NY) dataset
#   ./scripts/download-dimacs.sh all      # Download all datasets
#   ./scripts/download-dimacs.sh NY BAY   # Download specific datasets
#
# Available datasets:
#   NY   - New York (264K nodes, 730K edges) - smallest, good for quick tests
#   BAY  - San Francisco Bay (321K nodes, 800K edges)
#   COL  - Colorado (436K nodes, 1M edges)
#   FLA  - Florida (1.1M nodes, 2.7M edges)
#   NE   - Northeast USA (1.5M nodes, 3.9M edges)
#   CAL  - California/Nevada (1.9M nodes, 4.7M edges)
#   USA  - Full USA (23.9M nodes, 58.3M edges) - largest, ~335MB compressed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data"

BASE_URL="http://www.diag.uniroma1.it/challenge9/data/USA-road-d"

# Dataset definitions: name -> filename
declare -A DATASETS=(
    ["NY"]="USA-road-d.NY"
    ["BAY"]="USA-road-d.BAY"
    ["COL"]="USA-road-d.COL"
    ["FLA"]="USA-road-d.FLA"
    ["NE"]="USA-road-d.NE"
    ["CAL"]="USA-road-d.CAL"
    ["USA"]="USA-road-d.USA"
)

# Dataset sizes for user info
declare -A SIZES=(
    ["NY"]="~12MB compressed"
    ["BAY"]="~15MB compressed"
    ["COL"]="~20MB compressed"
    ["FLA"]="~50MB compressed"
    ["NE"]="~70MB compressed"
    ["CAL"]="~90MB compressed"
    ["USA"]="~335MB compressed"
)

download_dataset() {
    local name="$1"
    local filename="${DATASETS[$name]}"

    if [[ -z "$filename" ]]; then
        echo "Unknown dataset: $name"
        echo "Available: ${!DATASETS[*]}"
        return 1
    fi

    local gr_file="$DATA_DIR/${filename}.gr"
    local gz_file="$DATA_DIR/${filename}.gr.gz"
    local url="$BASE_URL/${filename}.gr.gz"

    if [[ -f "$gr_file" ]]; then
        echo "✓ $name already exists: $gr_file"
        return 0
    fi

    echo "Downloading $name (${SIZES[$name]})..."
    mkdir -p "$DATA_DIR"

    if command -v curl &> /dev/null; then
        curl -L -o "$gz_file" "$url"
    elif command -v wget &> /dev/null; then
        wget -O "$gz_file" "$url"
    else
        echo "Error: Neither curl nor wget found"
        return 1
    fi

    echo "Extracting..."
    gunzip "$gz_file"

    echo "✓ Downloaded: $gr_file"
}

list_datasets() {
    echo "Available DIMACS datasets:"
    echo ""
    printf "  %-6s %-25s %s\n" "Name" "Size" "Description"
    printf "  %-6s %-25s %s\n" "----" "----" "-----------"
    printf "  %-6s %-25s %s\n" "NY" "264K nodes, 730K edges" "New York (default)"
    printf "  %-6s %-25s %s\n" "BAY" "321K nodes, 800K edges" "San Francisco Bay"
    printf "  %-6s %-25s %s\n" "COL" "436K nodes, 1M edges" "Colorado"
    printf "  %-6s %-25s %s\n" "FLA" "1.1M nodes, 2.7M edges" "Florida"
    printf "  %-6s %-25s %s\n" "NE" "1.5M nodes, 3.9M edges" "Northeast USA"
    printf "  %-6s %-25s %s\n" "CAL" "1.9M nodes, 4.7M edges" "California/Nevada"
    printf "  %-6s %-25s %s\n" "USA" "23.9M nodes, 58.3M edges" "Full USA (largest)"
    echo ""
    echo "Currently downloaded:"
    if [[ -d "$DATA_DIR" ]]; then
        ls -la "$DATA_DIR"/*.gr 2>/dev/null || echo "  (none)"
    else
        echo "  (none)"
    fi
}

show_help() {
    echo "Download DIMACS road network datasets for benchmarking"
    echo ""
    echo "Usage:"
    echo "  $0                    Download default (NY) dataset"
    echo "  $0 all                Download all datasets (~260MB total)"
    echo "  $0 NY BAY COL         Download specific datasets"
    echo "  $0 --list             List available datasets"
    echo "  $0 --help             Show this help"
    echo ""
    echo "After downloading, run benchmarks with:"
    echo "  cargo bench"
    echo ""
    echo "Or run only DIMACS benchmarks:"
    echo "  cargo bench -- real_dimacs"
}

# Parse arguments
if [[ $# -eq 0 ]]; then
    # Default: download NY
    download_dataset "NY"
elif [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_help
elif [[ "$1" == "--list" || "$1" == "-l" ]]; then
    list_datasets
elif [[ "$1" == "all" ]]; then
    for name in NY BAY COL FLA NE CAL USA; do
        download_dataset "$name"
    done
else
    for name in "$@"; do
        download_dataset "$(echo "$name" | tr '[:lower:]' '[:upper:]')"
    done
fi

echo ""
echo "Run benchmarks with: cargo bench"
