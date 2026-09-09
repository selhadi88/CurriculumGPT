FROM node:20-alpine AS builder
WORKDIR /app

# pnpm via corepack, pinned. .npmrc sets node-linker=hoisted so vite/esbuild
# binaries resolve like a flat npm install; package.json's
# pnpm.onlyBuiltDependencies allow-lists esbuild's build script (otherwise
# pnpm 9.4+ fails the install with ERR_PNPM_IGNORED_BUILDS).
ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0
RUN corepack enable && corepack prepare pnpm@9.15.4 --activate

COPY package.json pnpm-lock.yaml .npmrc ./
RUN pnpm install --no-frozen-lockfile

COPY . .
ARG VITE_API_URL=http://localhost:7000
ENV VITE_API_URL=$VITE_API_URL

# Bundle with vite (type-check skipped for the image build).
RUN pnpm exec vite build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
