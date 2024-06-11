# Dockerfile for mongo-init
FROM mongo:5
COPY wait-for-mongo.sh /wait-for-mongo.sh
COPY init-replica.sh /init-replica.sh
RUN chmod +x /wait-for-mongo.sh /init-replica.sh
