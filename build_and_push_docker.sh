#!/bin/bash
# Manual Docker image building and publishing script
# This script builds and pushes Docker images to GitHub Container Registry

set -e

# Configuration
REGISTRY="ghcr.io"
IMAGE_NAME="computationalpathologygroup/polysome"
VERSION="${1:-latest}"  # Default to 'latest' if no version specified

echo "=== Polysome Docker Build & Push ==="
echo "Registry: ${REGISTRY}"
echo "Image: ${IMAGE_NAME}"
echo "Version: ${VERSION}"
echo ""

# Check if logged in to GitHub Container Registry
echo "Checking GitHub Container Registry login..."
if ! docker info 2>/dev/null | grep -q "${REGISTRY}"; then
    echo "Please login to GitHub Container Registry first:"
    echo "  docker login ${REGISTRY}"
    echo ""
    echo "Generate a Personal Access Token (classic) with 'write:packages' scope at:"
    echo "  https://github.com/settings/tokens"
    exit 1
fi

# Build GPU image
echo ""
echo "1. Building GPU Docker image..."
docker build -t ${REGISTRY}/${IMAGE_NAME}:${VERSION} -f Dockerfile .
echo "   ✓ GPU image built"

# Build ARM64 image
echo ""
echo "2. Building ARM64 Docker image..."
docker build -t ${REGISTRY}/${IMAGE_NAME}:${VERSION}-arm64 -f Dockerfile.arm64 --platform linux/arm64 .
echo "   ✓ ARM64 image built"

# Push images
echo ""
echo "3. Pushing images to registry..."
docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}
echo "   ✓ GPU image pushed"

docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}-arm64
echo "   ✓ ARM64 image pushed"

# Tag as latest if version is not latest
if [ "${VERSION}" != "latest" ]; then
    echo ""
    echo "4. Tagging as latest..."
    docker tag ${REGISTRY}/${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:latest
    docker tag ${REGISTRY}/${IMAGE_NAME}:${VERSION}-arm64 ${REGISTRY}/${IMAGE_NAME}:latest-arm64

    docker push ${REGISTRY}/${IMAGE_NAME}:latest
    docker push ${REGISTRY}/${IMAGE_NAME}:latest-arm64
    echo "   ✓ Tagged and pushed as latest"
fi

echo ""
echo "✓ Docker images published successfully!"
echo ""
echo "Images published:"
echo "  ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
echo "  ${REGISTRY}/${IMAGE_NAME}:${VERSION}-arm64"
if [ "${VERSION}" != "latest" ]; then
    echo "  ${REGISTRY}/${IMAGE_NAME}:latest"
    echo "  ${REGISTRY}/${IMAGE_NAME}:latest-arm64"
fi
echo ""
echo "Users can pull with:"
echo "  docker pull ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
echo "  docker pull ${REGISTRY}/${IMAGE_NAME}:${VERSION}-arm64"
