"""Tests for the rapid-mlx slot panel. They never stop a real serve process."""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import panel


def _request(port: int, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = response.read().decode()
            return response.status, json.loads(payload) if payload else {}
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode())
        finally:
            exc.close()


class SlotStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAPID_MLX_PANEL_STATE"] = self.tmp.name
        os.environ["RAPID_MLX_PANEL_NO_SEED"] = "1"
        os.environ["RAPID_MLX_PANEL_PORT"] = "8770"
        os.environ.pop("RAPID_MLX_BIN", None)
        panel._MODELS_CACHE["payload"] = None
        panel._MODELS_CACHE["at"] = 0
        panel._CATALOG_CACHE["payload"] = None
        panel._CATALOG_CACHE["at"] = 0
        panel._ENGINE_CACHE["payload"] = None
        panel._ENGINE_CACHE["at"] = 0

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_seed_matches_the_running_layout(self) -> None:
        os.environ["RAPID_MLX_PANEL_NO_SEED"] = "0"
        fresh = tempfile.TemporaryDirectory()
        os.environ["RAPID_MLX_PANEL_STATE"] = fresh.name
        slots = panel.load_slots()
        by_id = {slot["id"]: slot for slot in slots}
        self.assertEqual(set(by_id), {"smart", "fast", "uncensored", "vision"})
        self.assertEqual(by_id["fast"]["port"], 8004)
        self.assertEqual(by_id["fast"]["model"], "qwen3-4b-instruct-2507-4bit")
        self.assertEqual(by_id["fast"]["embedding_model"], "mlx-community/embeddinggemma-300m-8bit")
        self.assertEqual(by_id["fast"]["api_name"], "")
        argv = panel.build_argv(by_id["uncensored"])
        self.assertIn("orcarouter/Qwen3.8-27B-Uncensored-MLX-4bit", argv)
        self.assertIn("--gpu-memory-utilization", argv)
        self.assertIn("1", argv)
        config = argv[argv.index("--speculative-config") + 1]
        self.assertEqual(
            json.loads(config),
            {
                "method": "mtp",
                "num_speculative_tokens": 3,
                "model": "orcarouter/Qwen3.8-27B-Uncensored-MLX",
            },
        )
        self.assertIn("--mllm", panel.build_argv(by_id["vision"]))
        self.assertNotIn("--served-model-name", panel.build_argv(by_id["fast"]))
        self.assertEqual(by_id["fast"]["engine"], "rapid-mlx")
        self.assertNotIn("tensorfold", panel.build_argv(by_id["fast"])[0])
        fresh.cleanup()

    def test_api_name_is_a_stable_alias_for_the_weights(self) -> None:
        slot = panel._normalize_slot(
            {
                "id": "fast",
                "name": "fast",
                "port": 18004,
                "api_name": "fast",
                "model": "qwen3-4b-instruct-2507-4bit",
            }
        )
        argv = panel.build_argv(slot)
        self.assertEqual(argv[argv.index("--served-model-name") + 1], "fast")
        self.assertEqual(argv[2], "qwen3-4b-instruct-2507-4bit")

    def test_create_update_delete_persist(self) -> None:
        created = panel.create_slot(
            {"name": "lab", "port": 18010, "api_name": "lab", "model": "qwen3-4b-instruct-2507-4bit"}
        )
        self.assertEqual(created["id"], "lab")
        panel.update_slot("lab", {**created, "api_name": "snabb", "port": 18011})
        again = panel.load_slots()
        self.assertEqual(again[0]["api_name"], "snabb")
        self.assertEqual(again[0]["port"], 18011)
        panel.delete_slot("lab")
        self.assertEqual(panel.load_slots(), [])

    def test_reorder_persists_and_rejects_a_partial_list(self) -> None:
        panel.create_slot({"name": "one", "port": 18030, "model": "qwen3-4b-instruct-2507-4bit"})
        panel.create_slot({"name": "two", "port": 18031, "model": "qwen3-4b-instruct-2507-4bit"})
        panel.reorder_slots(["two", "one"])
        self.assertEqual([slot["id"] for slot in panel.load_slots()], ["two", "one"])
        with self.assertRaises(panel.PanelError):
            panel.reorder_slots(["one"])

    def test_same_port_and_name_can_be_saved(self) -> None:
        panel.create_slot({"name": "lab", "port": 18012, "model": "qwen3-4b-instruct-2507-4bit"})
        other = panel.create_slot({"name": "lab", "port": 18012, "model": "another-model"})
        self.assertNotEqual(other["id"], "lab")
        slots = panel.load_slots()
        self.assertEqual([slot["port"] for slot in slots], [18012, 18012])
        self.assertEqual([slot["name"] for slot in slots], ["lab", "lab"])

    def test_shared_port_is_busy_on_the_slot_that_does_not_own_it(self) -> None:
        first = panel.create_slot({"name": "a", "port": 18012, "model": "qwen3-4b-instruct-2507-4bit"})
        second = panel.create_slot({"name": "b", "port": 18012, "model": "another-model"})
        process = {
            "pid": 4242,
            "port": 18012,
            "command": "rapid-mlx serve one",
            "model": "one",
            "uptime": "1s",
        }
        runtime = {"pids": {first["id"]: {"pid": 4242, "port": 18012}}}
        enriched, unassigned = panel.classify(panel.load_slots(), runtime, [process])
        by_id = {slot["id"]: slot for slot in enriched}
        self.assertEqual(by_id[first["id"]]["status"], "running")
        self.assertEqual(by_id[second["id"]]["status"], "busy")
        self.assertEqual(unassigned, [])
        outsider = {"pids": {}}
        enriched, unassigned = panel.classify(panel.load_slots(), outsider, [process])
        statuses = [slot["status"] for slot in enriched]
        self.assertEqual(statuses, ["external", "busy"])
        self.assertEqual(unassigned, [])

    def test_extra_args_cannot_override_port_or_api_name(self) -> None:
        with self.assertRaises(panel.PanelError):
            panel._normalize_slot(
                {
                    "id": "lab",
                    "name": "lab",
                    "port": 18013,
                    "model": "qwen3-4b-instruct-2507-4bit",
                    "extra_args": "--port 9",
                }
            )

    def test_parse_live_serve_lines(self) -> None:
        orca = (
            "/Users/otto/.local/bin/rapid-mlx serve "
            "orcarouter/Qwen3.8-27B-Uncensored-MLX-4bit "
            '--speculative-config {"method":"mtp","model":"orcarouter/Qwen3.8-27B-Uncensored-MLX",'
            '"num_speculative_tokens":3} --gpu-memory-utilization 1 --port 8005'
        )
        parsed = panel.parse_serve_command(orca)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["port"], 8005)
        self.assertEqual(parsed["model"], "orcarouter/Qwen3.8-27B-Uncensored-MLX-4bit")
        wrapped = "/usr/bin/caffeinate -is /Users/otto/.local/bin/rapid-mlx serve qwen --port 8000"
        self.assertIsNone(panel.parse_serve_command(wrapped))
        script = 'Python -c if "serve" in line and rapid-mlx'
        self.assertIsNone(panel.parse_serve_command(script))
        folded = (
            "/Users/otto/.local/bin/tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit "
            "--drafter z-lab/Qwen3.8-27B-DFlash2 --name qwen38-27b --context 8192 "
            "--cortheon --learn --slide --port 8080"
        )
        parsed = panel.parse_serve_command(folded)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["engine"], "tensorfold")
        self.assertEqual(parsed["port"], 8080)
        self.assertEqual(parsed["model"], "TensorFold/Qwen3.8-27B-MLX-4bit")
        self.assertEqual(parsed["served_name"], "qwen38-27b")
        comfy = (
            "/Users/otto/ComfyUI/models/ComfyUI/ComfyUI/.venv/bin/python "
            "/Users/otto/ComfyUI/models/ComfyUI/ComfyUI/main.py --port 8192"
        )
        parsed = panel.parse_serve_command(comfy)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["engine"], "comfyui")
        self.assertEqual(parsed["port"], 8192)
        tracker = (
            "/Users/otto/ComfyUI/models/ComfyUI/ComfyUI/.venv/bin/python -c "
            "from multiprocessing.resource_tracker import main;main(17)"
        )
        self.assertIsNone(panel.parse_serve_command(tracker))

    def test_tensorfold_engine_keeps_its_flags_off_the_rapid_command(self) -> None:
        slot = panel._normalize_slot(
            {
                "id": "fold",
                "name": "fold",
                "port": 18018,
                "api_name": "qwen38-27b",
                "model": "TensorFold/Qwen3.8-27B-MLX-4bit",
                "engine": "tensorfold",
                "drafter": "z-lab/Qwen3.8-27B-DFlash2",
                "cortheon": True,
                "learn": True,
                "slide": True,
                "api_key": "test-slide-key",
                "context_length": 8192,
                "spec_decode": "mtp",
                "mtp_model": "should-not-appear",
            }
        )
        argv = panel.build_argv(slot)
        self.assertEqual(argv[0], panel.tensorfold_bin())
        self.assertEqual(argv[1], "serve")
        self.assertEqual(argv[2], "TensorFold/Qwen3.8-27B-MLX-4bit")
        self.assertEqual(argv[argv.index("--host") + 1], "127.0.0.1")
        self.assertEqual(argv[argv.index("--drafter") + 1], "z-lab/Qwen3.8-27B-DFlash2")
        self.assertEqual(argv[argv.index("--name") + 1], "qwen38-27b")
        self.assertEqual(argv[argv.index("--context") + 1], "8192")
        for flag in ("--cortheon", "--learn", "--slide", "--api-key-file"):
            self.assertIn(flag, argv)
        self.assertNotIn("test-slide-key", argv)
        self.assertTrue(argv[argv.index("--api-key-file") + 1].endswith("/keys/fold.key"))
        self.assertNotIn("--speculative-config", argv)
        self.assertNotIn("--mllm", argv)
        self.assertNotIn("should-not-appear", argv)
        advice = panel.describe_slot(slot, None)
        self.assertIn("model_type är qwen3", advice)
        self.assertIn("slide skriver om vikterna", advice)
        missing = panel._normalize_slot(
            {
                "id": "fold-open",
                "name": "fold-open",
                "port": 18020,
                "model": "TensorFold/Qwen3.8-27B-MLX-4bit",
                "engine": "tensorfold",
                "slide": True,
            }
        )
        self.assertEqual(missing["slide"], True)
        with self.assertRaises(panel.PanelError):
            panel.build_argv(missing)
        kept = panel._normalize_slot(
            {"id": "smart", "name": "smart", "port": 18019, "model": "qwen3.8-27b-4bit"}
        )
        self.assertEqual(kept["engine"], "rapid-mlx")
        self.assertFalse(kept["slide"])
        comfy = panel._normalize_slot(
            {"id": "comfy", "name": "comfy", "port": 18192, "engine": "comfyui", "model": ""}
        )
        argv = panel.build_argv(comfy)
        self.assertEqual(argv[0], panel.comfyui_python())
        self.assertTrue(argv[1].endswith("/main.py"))
        self.assertEqual(argv[argv.index("--listen") + 1], "127.0.0.1")
        self.assertEqual(argv[argv.index("--port") + 1], "18192")
        self.assertNotIn("serve", argv)
        self.assertIn("diffusionsservern", panel.describe_slot(comfy, None))
        optiq = panel._normalize_slot(
            {
                "id": "obliterated",
                "name": "obliterated",
                "port": 18010,
                "model": "mlx-community/Qwen3.8-27B-OBLITERATED-OptiQ-4bit",
                "engine": "optiq",
                "spec_decode": "mtp",
                "kv_cache_dtype": "int4",
                "max_tokens": 4096,
                "context_length": 32768,
                "api_key": "sk-optiq-panel",
            }
        )
        argv = panel.build_argv(optiq)
        self.assertEqual(argv[0], panel.optiq_bin())
        self.assertEqual(argv[1:3], ["serve", "--model"])
        self.assertIn("--mtp", argv)
        self.assertEqual(argv[argv.index("--kv-bits") + 1], "4")
        self.assertEqual(argv[argv.index("--max-context") + 1], "32768")
        self.assertEqual(argv[argv.index("--max-tokens") + 1], "4096")
        self.assertNotIn("--mllm", argv)
        self.assertNotIn("--speculative-config", argv)
        self.assertNotIn("sk-optiq-panel", argv)
        self.assertIn("OPTIQ_API_KEY", panel.describe_slot(optiq, None))
        heard = panel.parse_serve_command(
            "/Users/otto/.local/bin/optiq serve --model "
            "mlx-community/Qwen3.8-27B-OBLITERATED-OptiQ-4bit --host 127.0.0.1 --port 8010 --mtp"
        )
        self.assertIsNotNone(heard)
        assert heard is not None
        self.assertEqual(heard["engine"], "optiq")
        self.assertEqual(heard["port"], 8010)
        self.assertEqual(heard["model"], "mlx-community/Qwen3.8-27B-OBLITERATED-OptiQ-4bit")

    def test_turboquant_and_catalog_mtp_are_explicit_in_the_command(self) -> None:
        turbo = panel._normalize_slot(
            {
                "id": "lab",
                "name": "lab",
                "port": 18014,
                "model": "qwen3.6-35b-4bit",
                "kv_cache_turboquant": "k8v4",
                "spec_decode": "off",
            }
        )
        argv = panel.build_argv(turbo)
        self.assertEqual(argv[argv.index("--kv-cache-turboquant") + 1], "k8v4")
        self.assertIn("--no-spec-decode", argv)
        stopped = panel._normalize_slot(
            {
                "id": "lab",
                "name": "lab",
                "port": 18014,
                "model": "qwen3.6-35b-4bit",
                "kv_cache_turboquant": "off",
            }
        )
        argv = panel.build_argv(stopped)
        self.assertEqual(argv[argv.index("--kv-cache-turboquant") + 1], "none")
        catalog = panel._normalize_slot(
            {
                "id": "lab",
                "name": "lab",
                "port": 18014,
                "model": "qwen3.8-27b-4bit",
                "spec_decode": "mtp",
                "mtp_tokens_auto": True,
                "mtp_model": "",
            }
        )
        config = json.loads(
            panel.build_argv(catalog)[panel.build_argv(catalog).index("--speculative-config") + 1]
        )
        self.assertEqual(config, {"method": "mtp"})
        with self.assertRaises(panel.PanelError):
            panel._normalize_slot(
                {
                    "id": "lab",
                    "name": "lab",
                    "port": 18014,
                    "model": "qwen3.8-27b-4bit",
                    "kv_cache_dtype": "int8",
                    "kv_cache_turboquant": "v4",
                }
            )
        with self.assertRaises(panel.PanelError):
            panel._normalize_slot(
                {
                    "id": "lab",
                    "name": "lab",
                    "port": 18014,
                    "model": "qwen3-vl-4b-4bit",
                    "lane": "vision",
                    "spec_decode": "mtp",
                }
            )

    def test_advice_names_the_catalog_choice(self) -> None:
        qwen38 = {
            "hf_path": "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX",
            "mtp_draft_model": "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX",
            "mtp_speculative_tokens": 3,
            "mtp_default_enabled": True,
            "mtp_continuous_tier": "verified",
            "turboquant_tier": "unknown",
            "supports_native_mtp": False,
        }
        plain = panel._normalize_slot(
            {"id": "smart", "name": "smart", "port": 18015, "model": "qwen3.8-27b-4bit"}
        )
        text = panel.describe_slot(plain, qwen38)
        self.assertIn("bf16", text)
        self.assertIn("samma checkpoint, djup 3", text)
        qwen36 = {
            "hf_path": "mlx-community/Qwen3.6-35B-A3B-4bit",
            "mtp_draft_model": "mlx-community/Qwen3.6-35B-A3B-MTP-4bit",
            "mtp_speculative_tokens": 2,
            "mtp_default_enabled": True,
            "mtp_continuous_tier": "verified",
            "turboquant_tier": "k8v4_verified",
            "supports_native_mtp": False,
        }
        active = panel.describe_slot(
            panel._normalize_slot(
                {"id": "lab", "name": "lab", "port": 18016, "model": "qwen3.6-35b-4bit"}
            ),
            qwen36,
        )
        self.assertIn("bf16", active)
        self.assertIn("mlx-community/Qwen3.6-35B-A3B-MTP-4bit, djup 2", active)
        self.assertNotIn("k8v4", active)
        quiet = panel.describe_slot(
            panel._normalize_slot(
                {
                    "id": "lab",
                    "name": "lab",
                    "port": 18016,
                    "model": "qwen3.6-35b-4bit",
                    "spec_decode": "off",
                }
            ),
            qwen36,
        )
        self.assertIn("TurboQuant k8v4", quiet)
        self.assertIn("MTP är av", quiet)
        unmapped = panel.describe_slot(
            panel._normalize_slot(
                {
                    "id": "lab",
                    "name": "lab",
                    "port": 18017,
                    "model": "orcarouter/Qwen3.8-27B-Uncensored-MLX-4bit",
                    "spec_decode": "mtp",
                    "mtp_model": "",
                    "mtp_tokens_auto": True,
                }
            ),
            None,
        )
        self.assertIn("Ange ett", unmapped)

    def test_installed_catalog_declares_the_local_aliases(self) -> None:
        if panel.rapid_mlx_python() is None:
            self.skipTest("rapid-mlx python saknas")
        catalog = panel.load_catalog()
        self.assertEqual(
            catalog["qwen3.8-27b-4bit"]["mtp_draft_model"],
            "rapid-mlx/Qwen3.8-27B-4bit-MTP-MLX",
        )
        self.assertEqual(catalog["qwen3.8-27b-4bit"]["mtp_speculative_tokens"], 3)
        self.assertEqual(catalog["qwen3.6-35b-4bit"]["turboquant_tier"], "k8v4_verified")
        self.assertEqual(catalog["qwen3.6-35b-4bit"]["mtp_speculative_tokens"], 2)
        self.assertIsNone(catalog["qwen3-4b-instruct-2507-4bit"]["mtp_draft_model"])
        self.assertIsNone(catalog["qwen3-vl-4b-4bit"]["mtp_draft_model"])


class PanelHttpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAPID_MLX_PANEL_STATE"] = self.tmp.name
        os.environ["RAPID_MLX_PANEL_NO_SEED"] = "1"
        os.environ["RAPID_MLX_PANEL_PORT"] = "8770"
        os.environ.pop("RAPID_MLX_BIN", None)
        panel._MODELS_CACHE["payload"] = None
        panel._MODELS_CACHE["at"] = 0
        panel._CATALOG_CACHE["payload"] = None
        panel._CATALOG_CACHE["at"] = 0
        panel._ENGINE_CACHE["payload"] = None
        panel._ENGINE_CACHE["at"] = 0
        self.httpd = panel.ThreadingHTTPServer(("127.0.0.1", 0), panel.Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def test_page_and_crud(self) -> None:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=5) as response:
            html = response.read().decode()
        self.assertIn("Ny plats", html)
        self.assertIn("Förenklad vy", html)
        self.assertIn("weight-select", html)
        self.assertIn("mtp-select", html)
        self.assertIn("TurboQuant k8v4", html)
        self.assertIn("Katalogens djup", html)
        self.assertIn("TensorFold Zig", html)
        self.assertIn("drafter-select", html)
        self.assertIn("Cortheon", html)
        self.assertIn(">Slide<", html)
        self.assertIn("Egen vikt", html)
        self.assertNotIn("weight-list", html)
        self.assertIn("API-namn", html)
        status, created = _request(
            self.port,
            "POST",
            "/api/slots",
            {"name": "lab", "port": 18020, "api_name": "lab", "model": "qwen3-4b-instruct-2507-4bit"},
        )
        self.assertEqual(status, 201)
        status, updated = _request(
            self.port,
            "PUT",
            f"/api/slots/{created['id']}",
            {**created, "api_name": "snabb", "model": "qwen3.8-27b-4bit"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(updated["api_name"], "snabb")
        self.assertEqual(updated["model"], "qwen3.8-27b-4bit")
        status, state = _request(self.port, "GET", "/api/state")
        self.assertEqual(state["slots"][0]["api_name"], "snabb")
        status, deleted = _request(self.port, "DELETE", f"/api/slots/{created['id']}")
        self.assertEqual(deleted, {"ok": True})
        saved = json.loads((Path(self.tmp.name) / "slots.json").read_text())
        self.assertEqual(saved["slots"], [])
        status, preview = _request(
            self.port,
            "POST",
            "/api/preview",
            {
                "name": "lab",
                "port": 18021,
                "model": "qwen3-4b-instruct-2507-4bit",
                "kv_cache_turboquant": "off",
                "spec_decode": "off",
            },
        )
        self.assertEqual(status, 200, preview)
        self.assertIn("--kv-cache-turboquant", preview["command"])
        self.assertIn("none", preview["command"])
        self.assertIn("MTP är av", preview["advice"])

    def test_start_and_stop_use_the_saved_slot(self) -> None:
        script = Path(self.tmp.name) / "rapid-mlx"
        log = Path(self.tmp.name) / "args.txt"
        script.write_text(
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$RAPID_MLX_FAKE_LOG\"\nsleep 60\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        os.environ["RAPID_MLX_BIN"] = str(script)
        os.environ["RAPID_MLX_FAKE_LOG"] = str(log)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            slot_port = probe.getsockname()[1]
        status, created = _request(
            self.port,
            "POST",
            "/api/slots",
            {
                "name": "tillfallig",
                "port": slot_port,
                "api_name": "snabb",
                "model": "qwen3-4b-instruct-2507-4bit",
                "spec_decode": "off",
            },
        )
        self.assertEqual(status, 201, created)
        status, started = _request(self.port, "POST", f"/api/slots/{created['id']}/start")
        self.assertEqual(status, 200, started)
        text = log.read_text(encoding="utf-8")
        self.assertIn("qwen3-4b-instruct-2507-4bit", text)
        self.assertIn("--served-model-name snabb", text)
        self.assertIn(f"--port {slot_port}", text)
        self.assertIn("--no-spec-decode", text)
        status, state = _request(self.port, "GET", "/api/state")
        self.assertEqual(state["slots"][0]["status"], "running")
        status, stopped = _request(self.port, "POST", f"/api/slots/{created['id']}/stop")
        self.assertEqual(status, 200, stopped)
        self.assertTrue(stopped["stopped"])
        status, state = _request(self.port, "GET", "/api/state")
        self.assertEqual(state["slots"][0]["status"], "stopped")

    def test_second_slot_on_the_same_port_does_not_start(self) -> None:
        script = Path(self.tmp.name) / "rapid-mlx"
        script.write_text("#!/bin/sh\nsleep 60\n", encoding="utf-8")
        script.chmod(0o755)
        os.environ["RAPID_MLX_BIN"] = str(script)
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            slot_port = probe.getsockname()[1]
        body = {
            "port": slot_port,
            "model": "qwen3-4b-instruct-2507-4bit",
            "spec_decode": "off",
        }
        status, first = _request(self.port, "POST", "/api/slots", {"name": "delad", **body})
        self.assertEqual(status, 201, first)
        status, second = _request(self.port, "POST", "/api/slots", {"name": "delad", **body})
        self.assertEqual(status, 201, second)
        status, started = _request(self.port, "POST", f"/api/slots/{first['id']}/start")
        self.assertEqual(status, 200, started)
        status, state = _request(self.port, "GET", "/api/state")
        by_id = {slot["id"]: slot for slot in state["slots"]}
        self.assertEqual(by_id[first["id"]]["status"], "running")
        self.assertEqual(by_id[second["id"]]["status"], "busy")
        status, blocked = _request(self.port, "POST", f"/api/slots/{second['id']}/start")
        self.assertEqual(status, 400, blocked)
        status, spared = _request(self.port, "POST", f"/api/slots/{second['id']}/stop")
        self.assertEqual(status, 200, spared)
        self.assertFalse(spared["stopped"])
        status, state = _request(self.port, "GET", "/api/state")
        by_id = {slot["id"]: slot for slot in state["slots"]}
        self.assertEqual(by_id[first["id"]]["status"], "running")
        status, stopped = _request(self.port, "POST", f"/api/slots/{first['id']}/stop")
        self.assertEqual(status, 200, stopped)
        self.assertTrue(stopped["stopped"])


class EngineStatusTest(unittest.TestCase):
    def test_parse_releases_skips_prereleases_and_sorts(self) -> None:
        history = panel.parse_github_releases(
            [
                {
                    "tag_name": "v1.0.0-beta.9",
                    "prerelease": True,
                    "body": "beta",
                    "published_at": "2026-10-22T00:00:00Z",
                },
                {
                    "tag_name": "v1.1.9",
                    "prerelease": False,
                    "body": "# TensorFold 1.1.9\n\nSnabbare prompter.",
                    "published_at": "2026-10-29T11:00:00Z",
                    "html_url": "https://github.com/ashhart/TensorFold/releases/tag/v1.1.9",
                },
                {
                    "tag_name": "v1.1.1",
                    "prerelease": False,
                    "body": "Första.",
                    "published_at": "2026-10-19T11:33:00Z",
                    "html_url": "https://github.com/ashhart/TensorFold/releases/tag/v1.1.1",
                },
            ]
        )
        self.assertEqual([row["version"] for row in history], ["1.1.9", "1.1.1"])
        self.assertEqual(history[0]["summary"], "TensorFold 1.1.9")
        self.assertEqual(history[0]["published"], "2026-10-29")
        self.assertEqual(panel.version_status("1.1.1", "1.1.9"), "behind")
        self.assertEqual(panel.version_status("1.0.0", "0.16.10"), "ahead")
        self.assertEqual(panel.version_status("0.16.4", "0.16.4"), "current")
        self.assertEqual(panel.parse_version("rapid-mlx 0.16.4\n"), "0.16.4")

    def test_engine_status_uses_cache_and_binaries(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        os.environ["RAPID_MLX_PANEL_STATE"] = tmp.name
        os.environ["RAPID_MLX_PANEL_NO_SEED"] = "1"
        panel._ENGINE_CACHE["payload"] = None
        panel._ENGINE_CACHE["at"] = 0
        binary = Path(tmp.name) / "rapid-mlx"
        binary.write_text("#!/bin/sh\necho rapid-mlx 0.16.4\n", encoding="utf-8")
        binary.chmod(0o755)
        missing = Path(tmp.name) / "missing-tensorfold"
        os.environ["RAPID_MLX_BIN"] = str(binary)
        os.environ["TENSORFOLD_BIN"] = str(missing)
        calls = {"n": 0}

        def fake_fetch(url: str) -> list[dict]:
            calls["n"] += 1
            if "Rapid-MLX" in url:
                return [
                    {
                        "tag_name": "v1.0.0",
                        "prerelease": False,
                        "body": "Agent.",
                        "published_at": "2026-10-29T00:00:00Z",
                        "html_url": "https://github.com/raullenchai/Rapid-MLX/releases/tag/v1.0.0",
                    }
                ]
            return [
                {
                    "tag_name": "v1.1.9",
                    "prerelease": False,
                    "body": "Metal.",
                    "published_at": "2026-10-29T00:00:00Z",
                    "html_url": "https://github.com/ashhart/TensorFold/releases/tag/v1.1.9",
                }
            ]

        original = panel.fetch_github_releases
        original_pypi = panel.fetch_pypi_version
        panel.fetch_github_releases = fake_fetch
        panel.fetch_pypi_version = lambda package: "0.7.0"
        try:
            report = panel.engine_status(refresh=True)
            again = panel.engine_status(refresh=False)
        finally:
            panel.fetch_github_releases = original
            panel.fetch_pypi_version = original_pypi
            tmp.cleanup()
        self.assertEqual(calls["n"], 2)
        by_id = {row["id"]: row for row in report["engines"]}
        self.assertEqual(by_id["rapid-mlx"]["installed"], "0.16.4")
        self.assertEqual(by_id["rapid-mlx"]["latest"], "1.0.0")
        self.assertEqual(by_id["rapid-mlx"]["status"], "behind")
        self.assertEqual(by_id["rapid-mlx"]["history"][0]["summary"], "Agent.")
        self.assertIsNone(by_id["tensorfold"]["installed"])
        self.assertEqual(by_id["tensorfold"]["status"], "unknown")
        self.assertEqual(by_id["optiq"]["latest"], "0.7.0")
        self.assertEqual(again["engines"][0]["latest"], report["engines"][0]["latest"])

    def test_update_refuses_a_running_engine(self) -> None:
        with self.assertRaises(panel.PanelError):
            panel.assert_engine_idle("rapid-mlx", [{"engine": "rapid-mlx", "port": 8000}])

    def test_tensorfold_update_checks_the_archive_and_retargets_the_symlink(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        install = root / "install" / "bin"
        install.mkdir(parents=True)
        old = install / "tensorfold-native"
        old.write_text("#!/bin/sh\necho tensorfold-native 1.1.1\n", encoding="utf-8")
        old.chmod(0o755)
        link = root / "tensorfold"
        link.symlink_to(old)
        payload = root / "payload"
        nested = payload / "tensorfold-1.2.0-macos-arm64" / "bin"
        nested.mkdir(parents=True)
        binary = nested / "tensorfold-native"
        binary.write_text("#!/bin/sh\necho tensorfold-native 1.2.0\n", encoding="utf-8")
        binary.chmod(0o755)
        archive = root / "bundle.tar.gz"
        with panel.tarfile.open(archive, "w:gz") as bundle:
            bundle.add(payload / "tensorfold-1.2.0-macos-arm64", arcname="tensorfold-1.2.0-macos-arm64")
        digest = panel.hashlib.sha256(archive.read_bytes()).hexdigest()

        def fake_download(url: str, dest: Path) -> None:
            if url.endswith(".sha256"):
                dest.write_text(f"{digest}  tensorfold-1.2.0-macos-arm64.tar.gz\n", encoding="utf-8")
            else:
                dest.write_bytes(archive.read_bytes())

        original = panel.download_file
        panel.download_file = fake_download
        try:
            installed = panel.install_tensorfold_release("1.2.0", link)
        finally:
            panel.download_file = original
        self.assertEqual(panel.installed_engine_version(str(link)), "1.2.0")
        self.assertTrue(installed.endswith("tensorfold-1.2.0-macos-arm64/bin/tensorfold-native"))
        self.assertTrue(link.is_symlink())
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
