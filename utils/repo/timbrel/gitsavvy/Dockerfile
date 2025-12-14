FROM python:3.7


RUN mkdir -p ~/.pip && touch ~/.pip/pip.conf
RUN echo "[global]" >> ~/.pip/pip.conf && echo "[install]" >> ~/.pip/pip.conf

# 安装必要的系统依赖
RUN apt-get update && apt-get install -y curl

RUN pip install pytest

RUN pip install pipdeptree

RUN mkdir -p /repo && git config --global --add safe.directory /repo
        