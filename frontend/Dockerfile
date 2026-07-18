FROM node:20-alpine AS builder
WORKDIR /app

# Use npm (not pnpm) inside Docker — pnpm's symlinked store makes binaries like
# vite hard to resolve in a container. npm installs flat, reliable node_modules.
ENV NODE_ENV=development
COPY package.json ./
RUN npm install --include=dev

COPY . .
ARG VITE_API_URL=http://localhost:7000
ENV VITE_API_URL=$VITE_API_URL

# Bundle with vite (type-check skipped for the image build).
RUN npx vite build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
