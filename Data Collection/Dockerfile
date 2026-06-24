# ==========================================
# Stage 1: Build usque 
# ==========================================
FROM golang:1.24.1 AS builder-usque
RUN apt-get update && apt-get install -y git
WORKDIR /app
RUN git clone https://github.com/tungns1207/usque.git .
RUN go mod download
RUN go build -o usque -ldflags="-s -w" .

# ==========================================
# Stage 2: Build masque-plus 
# ==========================================
FROM golang:1.24.1 AS builder-masque
RUN apt-get update && apt-get install -y git
WORKDIR /app
RUN git clone https://github.com/ircfspace/masque-plus.git .
RUN go mod download
RUN go build -o masque-plus .

# ==========================================
# Stage 3: Final Image
# ==========================================
FROM ubuntu:22.04

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Update and install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /workspace

# Copy compiled binaries from builder stages and grant execution permissions
COPY --from=builder-usque /app/usque /workspace//usque
COPY --from=builder-masque /app/masque-plus /workspace//masque-plus
RUN chmod +x /workspace/usque /workspace/masque-plus

# Copy the files from your host machine into the container
COPY ./copy_folder/ /workspace/

# Run scripts
RUN chmod +x /workspace/container_install.sh
RUN ./container_install.sh

# Command to run when the container starts
# Command to run when the container starts (Giữ container sống chờ lệnh)
CMD ["tail", "-f", "/dev/null"]