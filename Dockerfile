# Forge GPU — Docker build
# Usage:
#   docker build -t forge-gpu .
#   docker run --gpus all forge-gpu run examples/particle-rain.toml
#   docker run --gpus all -p 8080:8080 -p 8081:8081 forge-gpu run examples/dam-break.toml --serve 8080

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y curl build-essential pkg-config && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
COPY . .

RUN cargo build --release -p forge-manifest

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

COPY --from=builder /app/target/release/forge /usr/local/bin/forge
COPY --from=builder /app/examples /examples

WORKDIR /examples
ENTRYPOINT ["forge"]
CMD ["run", "particle-rain.toml"]
