#!/usr/bin/env bash
uwsgi --ini /etc/uwsgi/uwsgi.ini &
/usr/bin/tf_serving_entrypoint.sh