FROM node:18-slim
WORKDIR /app
COPY . /app
RUN npm install -g http-server
EXPOSE 8080
CMD ["http-server", ".", "-p", "8080"]
