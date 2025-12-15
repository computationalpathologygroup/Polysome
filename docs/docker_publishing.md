# Docker Image Publishing Guide

This guide explains how to manually build and publish Docker images for Polysome.

## Prerequisites

1. **Docker installed** with BuildKit support
2. **GitHub Personal Access Token** with `write:packages` scope
   - Generate at: https://github.com/settings/tokens
   - Select "Generate new token (classic)"
   - Enable `write:packages` and `read:packages` scopes

## One-Time Setup

Login to GitHub Container Registry:

```bash
docker login ghcr.io
# Username: your-github-username
# Password: your-personal-access-token
```

## Publishing Docker Images

### Quick Publish (Automated Script)

```bash
# Build and push with version tag
./build_and_push_docker.sh v0.1.0

# Build and push as latest
./build_and_push_docker.sh latest
```

This script:
1. Builds GPU image (from `Dockerfile`)
2. Builds ARM64 CPU image (from `Dockerfile.arm64`)
3. Pushes both to GitHub Container Registry
4. Tags as `latest` if version is specified

### Manual Build and Push

If you prefer manual control:

```bash
# Set version
VERSION="v0.1.0"
IMAGE_NAME="ghcr.io/computationalpathologygroup/polysome"

# Build GPU image
docker build -t ${IMAGE_NAME}:${VERSION} -f Dockerfile .
docker push ${IMAGE_NAME}:${VERSION}

# Build ARM64 image (requires QEMU/Buildx for cross-platform)
docker buildx build \
  --platform linux/arm64 \
  -t ${IMAGE_NAME}:${VERSION}-arm64 \
  -f Dockerfile.arm64 \
  --push .

# Tag as latest
docker tag ${IMAGE_NAME}:${VERSION} ${IMAGE_NAME}:latest
docker push ${IMAGE_NAME}:latest
```

## Testing Before Publishing

Always test images locally before publishing:

```bash
# Test GPU image
docker run --rm \
  --gpus all \
  -v ./data:/data \
  -v ./output:/output \
  -v ./workflows:/workflows \
  -v ./prompts:/prompts \
  -e WORKFLOW_PATH=/workflows/test_workflow.json \
  ghcr.io/computationalpathologygroup/polysome:v0.1.0

# Test ARM64 image (on ARM64 system or with QEMU)
docker run --rm \
  -v ./data:/data \
  -v ./output:/output \
  -v ./workflows:/workflows \
  -v ./prompts:/prompts \
  -e WORKFLOW_PATH=/workflows/test_workflow.json \
  ghcr.io/computationalpathologygroup/polysome:v0.1.0-arm64
```

## Release Workflow

1. **Update version** in `pyproject.toml` and `src/polysome/__init__.py`
2. **Test locally**: Build and test Docker images
3. **Commit and tag**:
   ```bash
   git add .
   git commit -m "Release v0.1.0"
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin main
   git push origin v0.1.0
   ```
4. **Publish Docker images**:
   ```bash
   ./build_and_push_docker.sh v0.1.0
   ```
5. **Create GitHub Release**: This triggers PyPI publishing automatically
6. **Verify**: Check packages on PyPI and ghcr.io

## Published Image Locations

After publishing, images are available at:

- **GPU**: `ghcr.io/computationalpathologygroup/polysome:latest`
- **ARM64**: `ghcr.io/computationalpathologygroup/polysome:latest-arm64`
- **Versioned**: `ghcr.io/computationalpathologygroup/polysome:v0.1.0`

Users can pull with:
```bash
docker pull ghcr.io/computationalpathologygroup/polysome:latest
```

## Troubleshooting

### Build fails with "permission denied"
- Ensure Docker daemon is running
- Check Docker BuildKit is enabled: `export DOCKER_BUILDKIT=1`

### Push fails with "unauthorized"
- Re-login: `docker login ghcr.io`
- Verify token has `write:packages` scope

### ARM64 build fails
- Install QEMU: `docker run --rm --privileged multiarch/qemu-user-static --reset -p yes`
- Use Docker Buildx: `docker buildx create --use`

### Image too large
- This is normal for CUDA images (several GB)
- Ensure good internet connection for pushing
- Consider splitting into base and application images if needed
