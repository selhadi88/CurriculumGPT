FROM node:20-alpine AS builder
WORKDIR /app

# pnpm pinned via package.json "packageManager"; corepack activates that exact
# version. .npmrc sets node-linker=hoisted so vite/esbuild binaries resolve like
# a flat npm install; pnpm-workspace.yaml allow-lists esbuild's build script so
# pnpm does not fail with ERR_PNPM_IGNORED_BUILDS.
ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml .npmrc ./
RUN corepack enable && pnpm install --no-frozen-lockfile

COPY . .
ARG VITE_API_URL=http://localhost:7000
ENV VITE_API_URL=$VITE_API_URL

# Bundle with vite (type-check skipped for the image build).
RUN pnpm exec vite build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
