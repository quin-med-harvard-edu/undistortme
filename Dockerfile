# undistortme runtime image
#
# Contains: undistortme (MIT), dcm2niix (BSD), slicenii (built from source),
# and the FSL components needed for TOPUP.
#
# NOTE ON LICENSING: FSL is installed at build time from the University of
# Oxford's own conda channel and is licensed for NON-COMMERCIAL use only
# (https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html). Commercial users must
# obtain an FSL licence from Oxford University Innovation.

# Stage 1: Build slicenii + combinenii from the 0.2.2 source.
# The pre-built 0.2.2 release binary was compiled from 0.2.0 source (which
# hard-codes axis=0 with no axis-guessing).  Building from source ensures we
# get the axis-guessing fix that was added in the 0.2.2 codebase.
FROM rust:slim AS slicenii-builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch 0.2.2 \
        https://github.com/liamtimms/slicenii.git /src/slicenii \
    && cd /src/slicenii \
    && cargo build --release

FROM mambaorg/micromamba:1.5-noble

LABEL org.opencontainers.image.title="undistortme" \
      org.opencontainers.image.description="Susceptibility distortion correction for multi-echo EPI via FSL TOPUP" \
      org.opencontainers.image.source="https://github.com/liamtimms/undistortme" \
      org.opencontainers.image.licenses="MIT AND LicenseRef-FSL-NonCommercial"

# FSL pieces (topup/applytopup, fslmerge/fslmaths) from Oxford's channel,
# dcm2niix + python from conda-forge.
RUN micromamba install -y -n base \
      -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ \
      -c conda-forge \
      python=3.12 pip \
      fsl-topup fsl-avwutils fsl-data_standard \
      dcm2niix \
    && micromamba clean -a -y

ENV PATH=/opt/conda/bin:$PATH \
    FSLDIR=/opt/conda \
    FSLOUTPUTTYPE=NIFTI

# slicenii + combinenii — copied from the build stage
USER root
COPY --from=slicenii-builder --chmod=0755 /src/slicenii/target/release/slicenii /usr/local/bin/slicenii
COPY --from=slicenii-builder --chmod=0755 /src/slicenii/target/release/combinenii /usr/local/bin/combinenii

# The package itself
COPY --chown=mambauser:mambauser . /src/undistortme
# [test] extra included so the image can self-verify (see image.yml smoke step);
# [denoise] brings dipy so --denoise mppca works in the image
RUN pip install --no-cache-dir "/src/undistortme[test,denoise]"

USER mambauser
WORKDIR /data
ENTRYPOINT ["undistortme"]
CMD ["--help"]
