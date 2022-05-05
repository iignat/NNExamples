FROM tensorflow/serving

MAINTAINER Ignat Ikryanov <iignat@mail.ru>

ENV TZ=Europe/Moscow

ENV MODEL_NAME=fishdetect_model

COPY ./requirements.txt /var/requirements.txt 
COPY ./start.sh   /tmp/start.sh
ADD frontend /var/www/srv
ADD models/fishdetect_model.tar.gz models/
ADD conf /etc/uwsgi
WORKDIR /var/www/srv

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt-get update && apt-get install -y --no-install-recommends \
    uwsgi-plugin-python3 \
    python3.5 \
    python3-pip \
    python3-setuptools \
    python3-pil \
    python3-pil.imagetk \
    && \
    apt-get clean && \
    tr -d '\r' </tmp/start.sh >/var/start.sh && \
    ln -s /usr/lib/uwsgi/plugins/python3_plugin.so /usr/lib/uwsgi/plugins/python_plugin.so && \
    pip3 install -r /var/requirements.txt && \
    chmod +x /var/start.sh && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/var/start.sh"]