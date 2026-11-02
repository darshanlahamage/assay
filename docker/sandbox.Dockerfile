# Image for both the agent container and the grader container.
# Build: assay build-image   (or: docker build -f docker/sandbox.Dockerfile -t assay-sandbox:0.2 docker/)

FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates curl procps strace make \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "pytest>=8" "hypothesis>=6"

# The agent works as a normal user, never root.
RUN useradd --create-home --uid 1000 --shell /bin/bash agent \
 && mkdir -p /repo && chown agent:agent /repo \
 && git config --system user.name "agent" \
 && git config --system user.email "agent@sandbox.local" \
 && git config --system init.defaultBranch main \
 && git config --system safe.directory '*'

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /repo
CMD ["sleep", "infinity"]
