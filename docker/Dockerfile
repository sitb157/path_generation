FROM tensorflow/tensorflow

ARG USER_NAME
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd ${USER_NAME} --gid ${USER_ID}\
    && useradd -l -m ${USER_NAME} -u ${USER_ID} -g ${USER_ID} -s /bin/bash

USER root

RUN apt-get update 
RUN apt-get install -y \
    vim \ 
    git

# Install rmf
RUN apt-get update && apt-get install -y \
    python3-pip \
