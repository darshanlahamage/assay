# Allow-list HTTP(S) proxy for agent containers. Every request is logged.
# Build: assay build-image   (or: docker build -t assay-proxy:0.1 docker/proxy/)
FROM alpine:3.20
RUN apk add --no-cache tinyproxy
COPY tinyproxy.conf /etc/tinyproxy/tinyproxy.conf
COPY allowlist /etc/tinyproxy/allowlist
EXPOSE 8888
CMD ["tinyproxy", "-d", "-c", "/etc/tinyproxy/tinyproxy.conf"]
