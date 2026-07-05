#!/usr/bin/env bash
# Submit the training job to Azure ML and stream logs.
# Run from the project root: bash cloud/azure/submit.sh

set -euo pipefail

RESOURCE_GROUP="gpt2-finetune-rg"
WORKSPACE="gpt2-finetune-ws"

cd "$(dirname "$0")/../.."   # ensure project root is CWD

echo "Submitting job..."
JOB_NAME=$(az ml job create \
    --file cloud/azure/job.yml \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE" \
    --query name -o tsv)

echo ""
echo "Job submitted: $JOB_NAME"
echo "Studio URL  : $(az ml job show \
    --name "$JOB_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE" \
    --query 'services.Studio.endpoint' -o tsv)"
echo ""
echo "Streaming logs (Ctrl-C to detach, job keeps running)..."
az ml job stream \
    --name "$JOB_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE"

echo ""
echo "To download the checkpoint after completion:"
echo "  az ml job download --name $JOB_NAME --output-name checkpoints \\"
echo "    --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE \\"
echo "    --download-path ./checkpoints"
