FROM tensorflow/tensorflow:2.17.0-gpu-jupyter

LABEL maintainer="lucasbanunes"
LABEL email="lucasbanunes@gmail.com"

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

WORKDIR /workspace

# RUN groupadd -g ${GID} ${GROUPNAME} && \
#     useradd -m -u ${UID} -g ${GROUPNAME} ${USERNAME} && \
#     usermod -aG sudo ${USERNAME} && \
#     usermod -aG root ${USERNAME}

# USER ${USERNAME}
# WORKDIR /home/${USERNAME}
