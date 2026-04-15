# syntax=docker/dockerfile:1
#
# Builds att3ntion with compiled CUDA extensions.
#
# Base image URLs are passed as build args by the reusable workflow in
# astera/att3ntion-builder (private).  For local builds, supply them
# explicitly:
#
#   docker build \
#       --build-arg BASE_BUILDER=<builder-image> \
#       --build-arg BASE_RUNTIME=<runtime-image> .
# ---------------------------------------------------------------------------

ARG BASE_BUILDER
ARG BASE_RUNTIME

# ---------------------------------------------------------------------------
# Compile CUDA extensions
# ---------------------------------------------------------------------------
FROM ${BASE_BUILDER} AS compile

WORKDIR /app

COPY . .

# CUDA_ARCHS controls which GPU architectures to compile native code for.
# PTX is always emitted for the highest arch for forward compatibility.
# Default covers: A100 (80), A10G/RTX3080 (86), L4/RTX4090 (89), H100 (90).
ARG CUDA_ARCHS="80 86 89 90"
RUN sed -i "s/seen_archs.add('89')/seen_archs.update('${CUDA_ARCHS}'.split())/" setup.py && \
    pip install . --no-build-isolation

# ---------------------------------------------------------------------------
# Runtime -- lean base with compiled extensions and experiment scripts
# ---------------------------------------------------------------------------
FROM ${BASE_RUNTIME}

WORKDIR /app

# The venv from the compile stage contains the base dependencies PLUS the
# freshly-compiled extension .so files.
COPY --from=compile /app/.venv /app/.venv

# Copy source tree for experiment scripts (demo.py, tests/, etc.)
COPY --from=compile /app /app

ENV PATH="/app/.venv/bin:$PATH"
