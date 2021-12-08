IN_PORT=1999
DST_PORT=1999

# For Generative - delta-ATOMIC
sudo docker run --name nginx_gen_atomic -v $PWD/nginx_configs/gen_atomic.conf:/etc/nginx/nginx.conf:ro -d -p $IN_PORT:$DST_PORT nginx

IN_PORT=2000
DST_PORT=2000

# For Generative - delta-ATOMIC
sudo docker run --name nginx_gen_atomic -v $PWD/nginx_configs/gen_atomic.conf:/etc/nginx/nginx.conf:ro -d -p $IN_PORT:$DST_PORT nginx