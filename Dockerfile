FROM debian:latest

#### Dependencies ####
RUN \
  apt update \
  && apt install -y --no-install-recommends \
    ca-certificates curl gcc-12 \
  && rm -rf /var/lib/apt/lists/*

#### BUILD ####
WORKDIR /root
RUN \
  curl -L https://github.com/spinicist/riesling/releases/download/v0.12/riesling-linux.tar.gz > riesling-linux.tar.gz \
  && tar -xzf riesling-linux.tar.gz \
  && rm riesling-linux.tar.gz

RUN mkdir /root/app
WORKDIR /root/app

# Define default command.
ENTRYPOINT ["/root/riesling"]
