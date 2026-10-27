FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43 AS mr-build
ARG MRSCALEMASTER_REVISION=0095d0d352bfa14e1b14881a70cb7ccaa2ed4996
ARG MAST3R_SLAM_REVISION=e6f4e3d474fad0e11f561482012be864ba8c3f17
ARG LIETORCH_REVISION=e7df86554156b36846008d8ddbcc4d8521a16554
ARG CNPY_REVISION=4e8810b1a8637695171ed346ce68f6984e585ef4
ARG TORCH_CUDA_ARCH_LIST="8.0+PTX;12.0+PTX"
ARG MAX_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive CUDA_HOME=/usr/local/cuda \
    PATH=/opt/mr/venv/bin:$PATH \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS} \
    PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-dev python3-venv build-essential cmake git curl ca-certificates \
        libeigen3-dev libsuitesparse-dev libopenblas-dev zlib1g-dev \
        libgl1 libglib2.0-0 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/mr/venv
RUN python -m pip install --upgrade pip==25.2 \
    && python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
COPY docker/requirements-mrscalemaster.txt /tmp/mrscalemaster.txt
RUN python -m pip install -r /tmp/mrscalemaster.txt

WORKDIR /opt/mr/src
RUN git init MR.ScaleMaster \
    && cd MR.ScaleMaster \
    && git remote add origin https://github.com/team-aprl/MR.ScaleMaster.git \
    && git config core.sparseCheckout true \
    && printf '/cores/\n/cpp/\n/cmake/\n/config/\n/requirements.txt\n/README.md\n/LICENSE*\n' > .git/info/sparse-checkout \
    && git fetch --filter=blob:none --depth 1 origin ${MRSCALEMASTER_REVISION} \
    && git checkout --detach FETCH_HEAD \
    && git init MASt3R-SLAM \
    && cd MASt3R-SLAM \
    && git remote add origin https://github.com/rmurai0610/MASt3R-SLAM.git \
    && git fetch --depth 1 origin ${MAST3R_SLAM_REVISION} \
    && git checkout --detach FETCH_HEAD \
    && git submodule update --init --recursive --depth 1
COPY docker/mast3r-build.patch /tmp/mast3r-build.patch
WORKDIR /opt/mr/src/MR.ScaleMaster/MASt3R-SLAM
RUN git apply --check /tmp/mast3r-build.patch && git apply /tmp/mast3r-build.patch
RUN git clone https://github.com/princeton-vl/lietorch.git /opt/mr/src/lietorch \
    && git -C /opt/mr/src/lietorch checkout ${LIETORCH_REVISION} \
    && git -C /opt/mr/src/lietorch submodule update --init --recursive \
    && python -m pip install --no-build-isolation --no-deps /opt/mr/src/lietorch
RUN python -m pip install --no-build-isolation --no-deps \
        ./thirdparty/mast3r/asmk ./thirdparty/mast3r/dust3r/croco/models/curope \
        ./thirdparty/in3d/thirdparty/pyimgui \
    && python -m pip install --no-build-isolation --no-deps \
        -e ./thirdparty/mast3r -e ./thirdparty/in3d -e . \
    && python -m pip check

WORKDIR /opt/mr/src/MR.ScaleMaster
ARG G2O_REVISION=ed2c608283ce44b29a68ccd75e1cdc10f7e1b52e
RUN mkdir -p cpp/thirdparty \
    && git clone https://github.com/RainerKuemmerle/g2o.git cpp/thirdparty/g2o \
    && git -C cpp/thirdparty/g2o checkout ${G2O_REVISION} \
    && git clone https://github.com/rogersce/cnpy.git cpp/thirdparty/cnpy \
    && git -C cpp/thirdparty/cnpy checkout ${CNPY_REVISION} \
    && cmake -S cpp/thirdparty/g2o -B /tmp/g2o-build \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/mr \
        -DG2O_BUILD_APPS=OFF -DG2O_BUILD_EXAMPLES=OFF -DG2O_USE_OPENGL=OFF \
        -DG2O_USE_LOGGING=OFF -DG2O_USE_CHOLMOD=ON -DBUILD_SHARED_LIBS=ON \
    && cmake --build /tmp/g2o-build --parallel ${MAX_JOBS} \
    && cmake --install /tmp/g2o-build \
    && cmake -S cpp -B /tmp/mr-build \
        -DCMAKE_BUILD_TYPE=Release -DUSE_SYSTEM_G2O=ON \
        -DCMAKE_PREFIX_PATH=/opt/mr -DCMAKE_BUILD_RPATH=/opt/mr/lib \
        -DPython3_EXECUTABLE=/opt/mr/venv/bin/python \
        -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" \
    && cmake --build /tmp/mr-build --parallel ${MAX_JOBS} --target g2o_multirobot \
    && LD_LIBRARY_PATH=/opt/mr/lib python -c 'from cores import g2o_multirobot; import torch, lietorch, mast3r_slam_backends, curope'

COPY docker/mr-checkpoints.sha256 /tmp/mr-checkpoints.sha256
RUN cd MASt3R-SLAM && mkdir -p checkpoints && cd checkpoints \
    && while read -r checksum filename; do \
        curl --fail --location --retry 3 --output "$filename" \
          "https://download.europe.naverlabs.com/ComputerVision/MASt3R/$filename"; \
       done < /tmp/mr-checkpoints.sha256 \
    && sha256sum --check /tmp/mr-checkpoints.sha256
RUN mkdir -p /opt/mr/share/doc/third-party/g2o \
    && cp /opt/mr/src/lietorch/LICENSE /opt/mr/share/doc/third-party/lietorch-LICENSE \
    && cp /opt/mr/src/MR.ScaleMaster/cpp/thirdparty/cnpy/LICENSE /opt/mr/share/doc/third-party/cnpy-LICENSE \
    && cp /opt/mr/src/MR.ScaleMaster/cpp/thirdparty/g2o/doc/license-*.txt /opt/mr/share/doc/third-party/g2o/
RUN find /opt/mr/src -type d \( -name .git -o -name build -o -name __pycache__ \) -prune -exec rm -rf '{}' + \
    && rm -rf /opt/mr/src/lietorch /opt/mr/src/MR.ScaleMaster/cpp/thirdparty \
    && python -m pip freeze > /opt/mr/installed-packages.txt

FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-devel@sha256:c74b2049d204a55233abace509fdb1f6bdbe2d169bee382ed2f06cac00ba76c4 AS aligned

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_DISABLE_NATIVE_JIT=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --gid "${APP_GID}" app \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home --no-log-init app \
    && chown app:app /app

COPY requirements.txt ./
COPY docker/wheels/ ./docker/wheels/
RUN python -m pip install --break-system-packages --no-cache-dir \
        -r requirements.txt

RUN python -m pip install --break-system-packages --no-cache-dir \
        git+https://github.com/facebookresearch/sam3.git@11dec2936de97f2857c1f76b66d982d5a001155d \
    && python -m pip check \
    && python -I -c "from sam3.model_builder import build_sam3_image_model; from importlib.resources import files; assert callable(build_sam3_image_model); assert files('sam3').joinpath('assets/bpe_simple_vocab_16e6.txt.gz').is_file()"

COPY --chown=app:app ltmem/ ./ltmem/
COPY --chown=app:app run.py ./
COPY --chown=app:app configs/ ./configs/
COPY --chown=app:app scripts/ ./scripts/

COPY LICENSE /usr/share/doc/lt-mem/
COPY docker/mast3r-build.patch docker/mr-checkpoints.sha256 /usr/share/doc/lt-mem/requirements/
ENV PYTHONPATH=/app PYTHON_DOTENV_DISABLED=1 LLM_PROVIDER=qwen
USER app
CMD ["bash"]

FROM aligned AS full
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0-pthread libusb-1.0-0 libcholmod5 libcxsparse4 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=mr-build /opt/mr/ /opt/mr/
ENV MRSCALEMASTER_ROOT=/opt/mr/src/MR.ScaleMaster \
    MRSCALEMASTER_PYTHON=/opt/mr/venv/bin/python \
    MRSCALEMASTER_LIBRARY_PATH=/opt/mr/lib
USER app
RUN python - <<'PYTHON'
import subprocess
from run import registration_environment
env = registration_environment()
subprocess.run([env["MRSCALEMASTER_PYTHON"], "-c",
                "import ltmem.registration.frontend, ltmem.registration.register"],
               env=env, check=True)
PYTHON
