from __future__ import annotations

import argparse
import json
import mimetypes
import re
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .store import AnnotationStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "nnUNet_raw" / "Dataset140_matsOverview_finetune139"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "instance_annotation_dataset142"
MAX_JSON_BYTES = 2 * 1024 * 1024


class InstanceAnnotationServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], dataset_root: Path, output_root: Path):
        super().__init__(address, InstanceAnnotationHandler)
        self.store = AnnotationStore(dataset_root, output_root)


class InstanceAnnotationHandler(BaseHTTPRequestHandler):
    server: InstanceAnnotationServer

    def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _png(self, data: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _static(self, relative: str) -> None:
        candidate = (STATIC_ROOT / relative).resolve()
        static_root = STATIC_ROOT.resolve()
        if static_root not in candidate.parents and candidate != static_root:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict[str, object]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length") from error
        if length <= 0 or length > MAX_JSON_BYTES:
            raise ValueError("Invalid request size")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON object expected")
        return payload

    @staticmethod
    def _case_route(path: str) -> tuple[str, str] | None:
        match = re.fullmatch(r"/api/cases/([A-Za-z0-9_-]+)(?:/([A-Za-z0-9_.-]+))?", path)
        if not match:
            return None
        return match.group(1), match.group(2) or "details"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/api/config":
                self._json(
                    {
                        "dataset_root": str(self.server.store.dataset_root),
                        "output_root": str(self.server.store.output_root),
                        "cases": self.server.store.cases(),
                        "tile_size": 512,
                        "format": "evo-instance-annotations-v1",
                    }
                )
                return
            route = self._case_route(path)
            if route:
                case_id, action = route
                query = parse_qs(parsed.query)
                if action == "details":
                    self._json(self.server.store.case_details(case_id))
                    return
                if action == "tile.png":
                    self._png(
                        self.server.store.render_tile(
                            case_id,
                            x=int(query.get("x", [0])[0]),
                            y=int(query.get("y", [0])[0]),
                            size=int(query.get("size", [512])[0]),
                            view=str(query.get("view", ["combined"])[0]),
                            selected_cell=(
                                int(query["selected"][0]) if query.get("selected") else None
                            ),
                        )
                    )
                    return
                if action == "thumbnail.png":
                    self._png(self.server.store.render_thumbnail(case_id))
                    return
                if action == "validate":
                    self._json(self.server.store.validate(case_id))
                    return
            if path in {"/", "/index.html"}:
                self._static("index.html")
                return
            if path.startswith("/static/"):
                self._static(path.removeprefix("/static/"))
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except KeyError as error:
            self._json({"error": str(error)}, HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self._json({"error": f"Internal error: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        try:
            route = self._case_route(path)
            if not route:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            case_id, action = route
            payload = self._body()
            if action == "cells":
                operation = str(payload.get("action", "create"))
                if operation == "create":
                    self._json(self.server.store.create_cell(case_id, payload.get("label")))
                    return
                if operation == "rename":
                    self._json(
                        self.server.store.rename_cell(
                            case_id, int(payload["cell_id"]), str(payload.get("label", ""))
                        )
                    )
                    return
                if operation == "delete":
                    self._json(self.server.store.delete_cell(case_id, int(payload["cell_id"])))
                    return
                raise ValueError(f"Unknown cell action: {operation}")
            if action == "stroke":
                raw_points = payload.get("points")
                if not isinstance(raw_points, list):
                    raise ValueError("points must be a list")
                points = [(float(point[0]), float(point[1])) for point in raw_points]
                self._json(
                    self.server.store.apply_stroke(
                        case_id,
                        points=points,
                        radius=int(payload.get("radius", 2)),
                        operation=str(payload.get("operation", "assign")),
                        target=str(payload.get("target", "foreground")),
                        cell_id=(int(payload["cell_id"]) if payload.get("cell_id") else None),
                    )
                )
                return
            if action == "fill":
                self._json(
                    self.server.store.apply_fill(
                        case_id,
                        x=int(payload["x"]),
                        y=int(payload["y"]),
                        operation=str(payload.get("operation", "assign")),
                        target=str(payload.get("target", "foreground")),
                        cell_id=(int(payload["cell_id"]) if payload.get("cell_id") else None),
                    )
                )
                return
            if action == "complete":
                self._json(self.server.store.set_completed(case_id, bool(payload.get("completed"))))
                return
            if action == "export":
                self._json(self.server.store.export(case_id))
                return
            raise ValueError(f"Unknown action: {action}")
        except KeyError as error:
            self._json({"error": str(error)}, HTTPStatus.NOT_FOUND)
        except (ValueError, TypeError, IndexError, json.JSONDecodeError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self._json({"error": f"Internal error: {error}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: object) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8778, type=int)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = InstanceAnnotationServer(
        (args.host, args.port), args.dataset_root.resolve(), args.output_root.resolve()
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"Instance annotation tool: {url}", flush=True)
    print(f"Cases: {len(server.store.records)}", flush=True)
    print(f"Output: {server.store.output_root}", flush=True)
    if args.open_browser:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()

