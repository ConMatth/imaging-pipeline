# x86_64 only: antspyx (defacing) publishes no Linux ARM wheels, and building it
# from source takes a long time. Runs emulated, so slower, on Apple Silicon.
FROM --platform=linux/amd64 python:3.12-slim

# dcm2niix and HeuDiConv (stage 3) are Python dependencies, installed from PyPI below.
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

ENTRYPOINT ["imaging-inventory"]
CMD ["--help"]
