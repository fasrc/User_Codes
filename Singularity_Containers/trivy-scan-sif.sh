#!/bin/sh
# Usage: TRIVY_CACHE_DIR=/dir/with/3GB/free trivy-scan-sif.sh TARGET.sif [trivy rootfs options]
#
# Writes the vulnerability table, then a package inventory (--list-all-pkgs), to stdout.
# Exits with Trivy's scan status (e.g. from --exit-code) when the report renders; fails if no report.
# Don't forward: --, --scanners, --format, --output, --insecure, --cacert, or options that skip the
# scan (--help, --download-db-only); they break the report or change what is trusted.
# The work dir holds a copy of the image's root filesystem: set TMPDIR to scratch for large images.

set -eu -o pipefail
target=$1; shift
mkdir -p "$TRIVY_CACHE_DIR"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# Trivy image: $TRIVY_SIF if set; else a local SIF keyed by a hash of $TRIVY_IMAGE (pin it by
# digest); else ./trivy.sif. Pulled if missing. The name is a cache key, not a provenance check.
if [ -z "${TRIVY_SIF:-}" ]; then
  if [ -n "${TRIVY_IMAGE:-}" ]; then
    key=$(printf %s "$TRIVY_IMAGE" | sha256sum); TRIVY_SIF=$PWD/trivy-${key%% *}.sif
  else
    TRIVY_SIF=$PWD/trivy.sif
  fi
  [ -e "$TRIVY_SIF" ] || singularity pull "$TRIVY_SIF" "${TRIVY_IMAGE:-docker://ghcr.io/aquasecurity/trivy:latest}"
fi

# Copy the primary root filesystem partition (exactly one, with a valid ID) out of the SIF as a bare squashfs
id=$(singularity sif list "$target" | awk -F'|' '
  /Squashfs\/\*System/ { n++; id = $1; gsub(/[ \t]/, "", id); if (id !~ /^[1-9][0-9]*$/) bad = 1 }
  END { if (n != 1 || bad) exit 1; print id }')
singularity sif dump "$id" "$target" > "$work/rootfs.sqfs"

# Scan once to a temporary JSON report, then print it as a table and a package inventory.
# User options come first so the options the script relies on override them.
# shellcheck disable=SC2016  # the inner script is single-quoted on purpose
singularity exec --containall --no-mount bind-paths --workdir "$work" \
  --bind /etc/resolv.conf:/etc/resolv.conf:ro \
  --bind /etc/pki/tls/certs/ca-bundle.crt:/host-ca.pem:ro --env SSL_CERT_FILE=/host-ca.pem \
  --mount "type=bind,src=$TRIVY_CACHE_DIR,dst=/cache" --env TRIVY_CACHE_DIR=/cache \
  --mount "type=bind,src=$work/rootfs.sqfs,dst=/scan,image-src=/" \
  "$TRIVY_SIF" sh -ec '
    # Trust only the host bundle: Go also loads every file in the default CA directories
    mkdir -p /tmp/empty-ca; export SSL_CERT_DIR=/tmp/empty-ca
    rc=0
    trivy rootfs "$@" --scanners vuln --list-all-pkgs --format json --output /tmp/report.json /scan || rc=$?
    [ -s /tmp/report.json ] || { echo "Trivy did not produce a report" >&2; [ "$rc" -ne 0 ] || rc=1; exit "$rc"; }
    trivy convert --format table /tmp/report.json
    echo; echo "Package inventory:"
    trivy convert --format template --template "{{range .}}{{if .Packages}}{{println}}{{.Target}}{{println}}{{range .Packages}}  {{.Name}} {{.Version}}{{if .Release}}-{{.Release}}{{end}} {{.FilePath}}{{println}}{{end}}{{end}}{{end}}" /tmp/report.json
    exit "$rc"
  ' sh "$@"
