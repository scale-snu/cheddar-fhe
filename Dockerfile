FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# https://apt.kitware.com/
RUN apt-get update && \
    apt-get install -y ca-certificates gpg wget && \
    test -f /usr/share/doc/kitware-archive-keyring/copyright || \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    test -f /usr/share/doc/kitware-archive-keyring/copyright || \
    rm /usr/share/keyrings/kitware-archive-keyring.gpg && \
    apt-get install -y kitware-archive-keyring && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy-rc main' | tee -a /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y cmake

RUN apt-get update && apt-get install -y \
    build-essential git libtommath-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /cheddar
COPY libcheddar.so.gz /cheddar/lib/libcheddar.so.gz
RUN gunzip /cheddar/lib/libcheddar.so.gz && \
    chmod 755 /cheddar/lib/libcheddar.so
COPY include /cheddar/include
COPY parameters /cheddar/unittest/parameters
COPY unittest /cheddar/unittest

ENV LD_LIBRARY_PATH=/cheddar/lib:$LD_LIBRARY_PATH

RUN mkdir -p /cheddar/unittest/build

WORKDIR /cheddar/unittest/build

RUN cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

CMD ["/cheddar/unittest/build/boot_test"]
