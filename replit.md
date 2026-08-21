# AI Smart Traffic Management

Python prototype that detects vehicles, estimates lane queues, and compares adaptive traffic signal policies in a Streamlit control room.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string

## Run the Python dashboard

- `streamlit run app.py --server.port 5000 --server.address 0.0.0.0 --server.headless true` — run the traffic control room
- `python -m compileall -q traffic_ai app.py` — validate Python syntax

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `app.py` — Streamlit dashboard entry point
- `traffic_ai/detection.py` — YOLOv8n + ByteTrack wrapper and MP4 iterator
- `traffic_ai/density_estimator.py` — lane ROI assignment, weighted density, and queue estimation
- `traffic_ai/signal_controller.py` — fixed-time and rule-based policies
- `traffic_ai/rl_agent.py` — tabular Q-learning policy
- `traffic_ai/simulator.py` and `traffic_ai/evaluate.py` — benchmark simulation and metrics
- `traffic_ai/demo_data.py` — deterministic synthetic tracks used for the first-run demo
- `requirements.txt` — Python runtime dependencies

## Architecture decisions

- The dashboard starts with deterministic synthetic traffic so it is useful before a video file or model weights are available.
- YOLO and OpenCV are imported lazily by the detection module; dashboard-only usage does not need to initialize the model.
- The simulator is intentionally lightweight and uses extracted weighted densities as demand inputs rather than requiring SUMO.

## Product

Users can tune four approach demands, inspect vehicle counts, weighted density, occupancy, queue length, and signal decisions, then benchmark fixed-time, rule-based, and Q-learning controllers.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- The Python tools module must be installed before pip packages in this Nix-based environment.
- YOLO video mode requires a readable MP4 and downloads/loads `yolov8n.pt` on first model use.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
