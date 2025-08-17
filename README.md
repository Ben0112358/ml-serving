# ml-serving

`ml-serving` is the model serving stage of the ML pipeline. Its purpose is to take trained ML models from `ml-training` and expose them on endpoints.  

This submodule forms a self-contained step in the larger ML pipeline:

```
ml-infra → ml-data → ml-training → ml-serving → ml-ui
```

`ml-serving` can be run **locally** for development or as part of the **full pipeline** orchestrated via `execute.sh` from https://github.com/Ben0112358/ml-pipeline.

To get an overview of how all sub-repos in the full pipeline are tied together, refer to https://github.com/Ben0112358/ml-meta. Links to all sub-repos can be found therein as well.

---

## 📁 Project Structure

```
ml-serving/
├── docker-compose.dummy_project.yaml   # Docker Compose file for containerized run
├── Dockerfile.dummy_project            # Dockerfile for containerized project
├── LICENSE
├── poetry.lock
├── pyproject.toml
├── README.md
├── src/
│   └── ml_serving/
│       ├── config.py                   # Global configuration (paths, suffixes)
│       ├── dummy_project/
│       │   ├── serving.py              # Core serving logic (API endpoints)
│       │   ├── utils/                  # Project-specific helpers
│       │   ├── __main__.py             # Local dev CLI entrypoint
│       │   └── __init__.py
│       └── utils/                      # Shared utils (logging, metrics, etc.)
└── tests/                              # Unit tests
```

---

## ✅ Prerequisites

- **OS**: Linux or macOS  
- **Docker**: Installed and running  
- **Python**: 3.12+  
- **Poetry**: For dependency management  

Set the base directory where shared ML assets and configs are stored:

```bash
export ML_HOMELAB_ROOT=/absolute/path/to/ml-homelab
```

Also set network and port for containerized serving:

```bash
export DOCKER_NETWORK_NAME=<network_name>
export SERVING_PORT=<port_number>
```

---

## 🐳 Containerized run (more control)

`ml-serving` can be run for example in the following way. You may add args as you see fit.

```bash
export ML_HOMELAB_ROOT=/path/to/ml_homelab_root
export DOCKER_NETWORK_NAME=<network_name>
export SERVING_PORT=<port_number>
docker-compose -f docker-compose.<project_name>.yaml -p "<project_name>_<mode>" build --no-cache
docker-compose -f docker-compose.<project_name>.yaml -p "<project_name>_<mode>" up
```

For more control, the following can be exported:

```bash
export MODEL_DIR=/path/to/models
export LOGS_DIR=/path/to/logs
export OUTPUT_SUFFIX=some_suffix
```

**Notes**:
- The serving container exposes the API on `localhost:$SERVING_PORT`.  
- This mode is a lightweight wrapper around docker-compose.<project_name>.yaml for convenience during development.

---

## 🐍 Python run (less control; simplified)

Run `ml-serving` locally with sensible defaults:

```bash
export ML_HOMELAB_ROOT=/path/to/ml_homelab_root
export DOCKER_NETWORK_NAME=<network_name>
export SERVING_PORT=<port_number>
python -m ml_serving.<project_name>
```

Or with more control over directories and outputs:

```bash
export ML_HOMELAB_ROOT=/path/to/ml_homelab_root
export DOCKER_NETWORK_NAME=<network_name>
export SERVING_PORT=<port_number>
export MODEL_DIR=/path/to/models
export LOGS_DIR=/path/to/logs
export OUTPUT_SUFFIX=some_suffix

python -m ml_serving.<project_name>
```
Also here, the serving container exposes the API on `localhost:$SERVING_PORT`.  

---

## ➕ Adding a New Project
1. Create a folder under `ml_serving/` with your project name:

```
src/ml_serving/<new_project>/
```

2. Implement the modules (mirroring `dummy_project`):

- `serving.py` → core serving logic (API endpoints)  
- `utils/` → project-specific utils
- `__main__.py` → optional CLI entrypoint for local dev  
- `__init__.py` → marks the package  

3. Add corresponding `docker-compose.<new_project>.yaml` and `Dockerfile.<new_project>`.

4. Set project-specific configuration in `ml_serving/config.py` or via environment variables (`MODEL_DIR`, `LOGS_DIR`, `OUTPUT_SUFFIX`, `DOCKER_NETWORK_NAME`, `SERVING_PORT`).  

`src/ml_serving/dummy_project` is a very simple project which can be studied to learn how it all ties together.

---

## 🧪 Testing

Run unit tests with Poetry:

```bash
poetry run pytest tests/
```
