#!/bin/bash
set -euxo pipefail

apt-get update
apt-get install -y build-essential git curl wget jq pkg-config ca-certificates unzip rsync
