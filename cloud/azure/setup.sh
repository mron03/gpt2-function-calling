#!/usr/bin/env bash
# One-time provisioning: resource group, Azure ML workspace, GPU compute cluster.
# Run from anywhere; all paths are absolute.
#
# Usage: bash cloud/azure/setup.sh [--subscription <id>]

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
RESOURCE_GROUP="gpt2-finetune-rg"
LOCATION="eastus"
WORKSPACE="gpt2-finetune-ws"
COMPUTE="gpt2-cluster"
VM_SIZE="Standard_NC4as_T4_v3"   # 1× NVIDIA T4 GPU, 4 vCPU, 28 GB RAM

# Optional: override subscription via --subscription <id>
if [[ "${1:-}" == "--subscription" ]]; then
    az account set --subscription "$2"
fi

SUBSCRIPTION=$(az account show --query id -o tsv)
echo "Subscription : $SUBSCRIPTION"
echo "Resource group: $RESOURCE_GROUP ($LOCATION)"
echo "Workspace     : $WORKSPACE"
echo "Compute       : $COMPUTE ($VM_SIZE)"
echo ""

# ── Azure ML extension ────────────────────────────────────────────────────────
echo "Ensuring azure-ai-ml extension is installed..."
az extension add --name azure-ai-ml --allow-preview true -y 2>/dev/null || \
    az extension update --name azure-ai-ml 2>/dev/null || true

# ── Resource group ────────────────────────────────────────────────────────────
echo "Creating resource group..."
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none

# ── Workspace ─────────────────────────────────────────────────────────────────
echo "Creating Azure ML workspace (this takes ~2 min)..."
az ml workspace create \
    --name "$WORKSPACE" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none

# ── GPU compute cluster ───────────────────────────────────────────────────────
echo "Creating GPU compute cluster..."
az ml compute create \
    --name "$COMPUTE" \
    --type AmlCompute \
    --size "$VM_SIZE" \
    --min-instances 0 \
    --max-instances 1 \
    --idle-time-before-scale-down 300 \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE" \
    --output none

echo ""
echo "Done. Next step:"
echo "  bash azure/submit.sh"
