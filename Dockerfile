FROM yyq465009551/torch112_cu113:swin3d_v2

RUN apt update \
 && apt install sudo

ARG USERNAME

# Create the user
RUN groupadd $USERNAME \
    && useradd -g $USERNAME -m $USERNAME \
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME

RUN pip install matplotlib scipy timm ptflops tensorboardX

ENV SHELL /bin/bash