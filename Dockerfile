FROM debian:latest

#### Dependencies ####
RUN \
  apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    pkg-config \
    software-properties-common \
    git man unzip wget curl cmake \
    ninja-build zip zlib1g zlib1g-dev autoconf autoconf-archive \
  && rm -rf /var/lib/apt/lists/*

#### BUILD ####
WORKDIR /root
RUN \
  git clone https://github.com/spinicist/riesling.git \
  && cd riesling \
  && export VCPKG_FORCE_SYSTEM_BINARIES=0 \
  && ./bootstrap.sh -i $PWD/../ -f native \
  && cd .. \
  && rm -rf riesling .cache

RUN mkdir /root/app
WORKDIR /root/app

# Define default command.
ENTRYPOINT ["/root/bin/riesling"]
