#!/bin/bash
# Export Criterion benchmark results to CSV for plotting
#
# Usage:
#   ./scripts/export-results.sh                    # Export all results
#   ./scripts/export-results.sh random_queries     # Export specific benchmark
#   ./scripts/export-results.sh --help             # Show help
#
# Output:
#   Writes CSV to stdout with columns: benchmark,variant,mean_ns,std_dev_ns
#
# Requirements:
#   - jq (JSON processor)
#   - Run 'cargo bench' first to generate results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CRITERION_DIR="$PROJECT_ROOT/target/criterion"

show_help() {
    echo "Export Criterion benchmark results to CSV"
    echo ""
    echo "Usage:"
    echo "  $0                         Export all benchmark results"
    echo "  $0 <benchmark_group>       Export specific benchmark group"
    echo "  $0 --list                  List available benchmark groups"
    echo "  $0 --help                  Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 > results.csv"
    echo "  $0 random_queries > random.csv"
    echo "  $0 dijkstra_rank > rank.csv"
    echo ""
    echo "Output format: benchmark,variant,mean_ns,std_dev_ns"
}

list_groups() {
    echo "Available benchmark groups:"
    if [[ -d "$CRITERION_DIR" ]]; then
        for dir in "$CRITERION_DIR"/*/; do
            name=$(basename "$dir")
            if [[ "$name" != "report" ]]; then
                echo "  $name"
            fi
        done
    else
        echo "  (none - run 'cargo bench' first)"
    fi
}

check_jq() {
    if ! command -v jq &> /dev/null; then
        echo "Error: jq is required but not installed" >&2
        echo "Install with: apt install jq (Debian/Ubuntu) or brew install jq (macOS)" >&2
        exit 1
    fi
}

export_group() {
    local group="$1"
    local group_dir="$CRITERION_DIR/$group"

    if [[ ! -d "$group_dir" ]]; then
        echo "Error: Benchmark group '$group' not found" >&2
        echo "Run 'cargo bench -- $group' first" >&2
        return 1
    fi

    local bench_name
    local estimates
    local mean
    local std_dev

    for bench_dir in "$group_dir"/*/; do
        bench_name=$(basename "$bench_dir")
        estimates="$bench_dir/new/estimates.json"

        if [[ -f "$estimates" ]]; then
            mean=$(jq -r '.mean.point_estimate' "$estimates")
            std_dev=$(jq -r '.std_dev.point_estimate' "$estimates")
            echo "$group,$bench_name,$mean,$std_dev"
        fi
    done
}

export_all() {
    local group
    for group_dir in "$CRITERION_DIR"/*/; do
        group=$(basename "$group_dir")
        if [[ "$group" != "report" && -d "$group_dir" ]]; then
            export_group "$group"
        fi
    done
}

# Parse arguments
if [[ $# -eq 0 ]]; then
    check_jq
    echo "benchmark,variant,mean_ns,std_dev_ns"
    export_all
elif [[ "$1" == "--help" || "$1" == "-h" ]]; then
    show_help
elif [[ "$1" == "--list" || "$1" == "-l" ]]; then
    list_groups
else
    check_jq
    echo "benchmark,variant,mean_ns,std_dev_ns"
    for group in "$@"; do
        export_group "$group"
    done
fi
