FROM python:3.10-slim

WORKDIR /swiftcfd

COPY ./requirements.txt .

# Install system dependencies (Debian-based)
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

ENV PETSC_CONFIGURE_OPTIONS="--download-fblaslapack=1"

RUN python3 -m venv /dev/venv
ENV PATH="/dev/venv/bin:$PATH"

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN rm ./requirements.txt