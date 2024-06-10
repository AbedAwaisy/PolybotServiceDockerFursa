# Dockerfile for mongo-init
FROM mongo:5
COPY init-mongo.js /docker-entrypoint-initdb.d/
COPY wait-for-mongo.sh /wait-for-mongo.sh
RUN chmod +x /wait-for-mongo.sh
