from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
import numpy as np

import datetime
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import zoneinfo
from .util import *
from .context_processors import ALGORITHM_ID2NAME, ALGORITHM_NAME2ID

import MAF.algorithms.preprocessing as preprocessing
import MAF.algorithms.inprocessing as inprocessing
import MAF.algorithms.postprocessing as postprocessing

from MAF.benchmark.crehate.crehate_demo import check_hatespeech
from MAF.benchmark.kobbq.kobbq_demo import check_korean_bias, KoBBQArguments
from MAF.benchmark.bbg.bbg_demo import BBGArguments, run_bbg_pipeline
from MAF.algorithms.inprocessing.kobbq_steer.demo import MAX_CHARS, compare_sample
from MAF.algorithms.inprocessing.deemc.demo import Lambda0Missing, compare_lambda
from MAF.algorithms.postprocessing.casual_path_tracing.CausalPathTracing import (
    DEFAULT_CPT_SAMPLES,
    MAX_CPT_SAMPLES,
    mitigate_cpt,
)
from MAF.algorithms.postprocessing.casual_path_tracing.VisionCausalPathTracing import (
    DEFAULT_VISION_SAMPLE,
    IMAGENET_VAL_SIZE,
    predict_vision_cpt,
)

from MAF.metric.latte.check_toxicity import check_toxicity


def _kobbq_steer_summary():
    path = os.environ["PYTHONPATH"] + "/MAF/result/kobbq_steer/qwen2.5-7b/summary.json"
    if not os.path.exists(path):
        return None, []
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        metric_keys = ["kobbq_amb_acc", "kobbq_dis_acc", "bias_amb", "bias_dis"]
        if "kmmlu_acc" in d["base"]:
            metric_keys += ["kmmlu_acc", "rel_drop"]
        summary = {"base": d["base"], "alphasteer": d["alphasteer"]}
        for cond in summary.values():
            missing = [k for k in metric_keys if k not in cond]
            if missing:
                raise KeyError(missing)
    except (OSError, ValueError, KeyError, TypeError) as e:
        logging.getLogger(__name__).warning("unreadable summary %s: %r", path, e)
        return None, []
    return summary, metric_keys


def _parse_cpt_indices(text):
    """'7, 3 ,7' → [3, 7]; 빈 문자열은 기본 샘플. 형식·범위 오류는 ValueError."""
    parts = [p.strip() for p in (text or "").split(",") if p.strip()]
    if not parts:
        return list(DEFAULT_CPT_SAMPLES)
    if not all(p.isdigit() for p in parts):
        raise ValueError("샘플 인덱스는 0 이상의 정수를 쉼표로 구분해 입력해 주세요.")
    indices = sorted({int(p) for p in parts})
    if len(indices) > MAX_CPT_SAMPLES:
        raise ValueError(f"샘플은 한 번에 {MAX_CPT_SAMPLES}개까지 입력할 수 있습니다.")
    if indices[-1] >= _KNOWN_1000_SIZE:
        raise ValueError(
            f"known_1000 인덱스는 0~{_KNOWN_1000_SIZE - 1} 사이여야 합니다."
        )
    return indices


_KNOWN_1000_SIZE = 1209
# 비전 추적(약 15분)은 백그라운드 스레드 1개로 직렬 실행; {idx: {"started", "error"}} 는 프로세스 수명 동안만 유지
_VISION_JOBS = {}
_VISION_LOCK = threading.Lock()


def _cpt_vision_cached(sample_idx):
    """저장된 run_* 중 이 샘플을 추적한 최신 것의 샘플별 지표; 없으면 None."""
    root = os.path.join(
        os.environ["PYTHONPATH"],
        "MAF",
        "result",
        "casual_path_tracing",
        "vit_tiny_patch16_224",
    )
    try:
        runs = sorted(d for d in os.listdir(root) if d.startswith("run_"))
    except OSError:
        return None
    for run in reversed(runs):
        path = os.path.join(root, run, "path_forward_results", "sample_wise_union.json")
        try:
            with open(path, encoding="utf-8") as f:
                r = json.load(f).get(str(sample_idx))
            if r is None:
                continue
            return {
                "run": run,
                "path_node_ratio": float(r["path_node_ratio"]),
                "decision_kept": bool(r["decision_kept"]),
                "path_prediction": int(r["path_decision"]),
            }
        except FileNotFoundError:
            continue  # 추적 중인 run 은 아직 결과 파일이 없다
        except (OSError, ValueError, KeyError, TypeError) as e:
            logging.getLogger(__name__).warning(
                "unreadable vision sample result %s: %r", path, e
            )
    return None


_KST = zoneinfo.ZoneInfo(
    "Asia/Seoul"
)  # 화면에 보이는 시각은 컨테이너 시계(UTC)가 아니라 KST
_VISION_TRACE_TIMEOUT = int(os.environ.get("CPT_VISION_TIMEOUT_SEC", 40 * 60))
_VISION_WATCHERS = {}
_TRACE_MODULE = (
    "MAF.algorithms.postprocessing.casual_path_tracing.VisionCausalPathTracing"
)


def _watch_vision_trace(sample_idx):
    """별도 프로세스로 추적(약 15분)을 돌리고 상한을 넘기면 죽인다.

    서브프로세스인 이유: 노이즈 탐색 루프에 상한이 없어 일부 샘플은 끝나지 않고, 스레드는 죽일 수 없다.
    CUDA 도 Django 프로세스 밖에 둬서 서버 재시작과 분리한다.
    """
    cmd = [
        sys.executable,
        "-c",
        f"from {_TRACE_MODULE} import trace_vision_cpt; trace_vision_cpt({sample_idx})",
    ]
    try:
        with _VISION_LOCK:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            try:
                code = proc.wait(timeout=_VISION_TRACE_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                _VISION_JOBS[sample_idx]["error"] = (
                    f"시간 초과 ({_VISION_TRACE_TIMEOUT // 60}분)"
                )
                return
        if code != 0:
            _VISION_JOBS[sample_idx]["error"] = f"종료 코드 {code}"
            return
    except OSError as e:
        logging.getLogger(__name__).exception(
            "vision trace could not start for %s", sample_idx
        )
        _VISION_JOBS[sample_idx]["error"] = f"프로세스 시작 실패 ({type(e).__name__})"
        return
    _VISION_JOBS.pop(sample_idx, None)


_ADOPT_POLL_SEC = 10


def _find_trace_pid(sample_idx):
    """서버 재시작으로 고아가 된 추적 프로세스를 /proc 에서 찾는다(Linux)."""
    marker = f"trace_vision_cpt({sample_idx})".encode()
    try:
        pids = [p for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return None
    for pid in pids:
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if marker in f.read():
                    return int(pid)
        except OSError:
            continue
    return None


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _process_elapsed(pid):
    """프로세스가 시작된 뒤 지난 초(/proc/<pid>/stat 의 starttime 기준)."""
    with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
        fields = f.read().rsplit(")", 1)[1].split()
    start_ticks = int(fields[19])  # 22번째 필드(starttime), ')' 뒤 기준으로 20번째
    with open("/proc/uptime", encoding="utf-8") as f:
        uptime = float(f.read().split()[0])
    return uptime - start_ticks / os.sysconf("SC_CLK_TCK")


def _adopt_vision_trace(sample_idx, pid):
    """이전 서버 세션이 띄운 추적을 다시 감시한다: 남은 상한까지 기다리고, 넘기면 죽인다."""
    try:
        elapsed = _process_elapsed(pid)
    except (OSError, ValueError, IndexError):
        elapsed = 0.0
    started = datetime.datetime.now(_KST) - datetime.timedelta(seconds=elapsed)
    _VISION_JOBS[sample_idx] = {
        "started": started.strftime("%Y-%m-%d %H:%M"),
        "error": None,
    }
    deadline = time.monotonic() + max(0.0, _VISION_TRACE_TIMEOUT - elapsed)

    def _watch():
        while _pid_alive(pid):
            if time.monotonic() >= deadline:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
                _VISION_JOBS[sample_idx]["error"] = (
                    f"시간 초과 ({_VISION_TRACE_TIMEOUT // 60}분)"
                )
                return
            time.sleep(_ADOPT_POLL_SEC)
        _VISION_JOBS.pop(sample_idx, None)

    t = threading.Thread(target=_watch, daemon=True)
    _VISION_WATCHERS[sample_idx] = t
    t.start()


def _start_vision_trace(sample_idx):
    _VISION_JOBS[sample_idx] = {
        "started": datetime.datetime.now(_KST).strftime("%Y-%m-%d %H:%M"),
        "error": None,
    }
    t = threading.Thread(target=_watch_vision_trace, args=(sample_idx,), daemon=True)
    _VISION_WATCHERS[sample_idx] = t
    t.start()


def _parse_vision_index(text):
    text = (text or "").strip()
    if not text:
        return DEFAULT_VISION_SAMPLE
    if not text.isdigit() or int(text) >= IMAGENET_VAL_SIZE:
        raise ValueError(
            f"ImageNet val 인덱스는 0~{IMAGENET_VAL_SIZE - 1} 사이의 정수 하나여야 합니다."
        )
    return int(text)


_CPT_KEYS = ("path_node_ratio", "decision_kept_ratio")
# pythia-14m 은 작지만 result/ 디렉터리 생성·root logger 조작을 겹치지 않게 한다
_CPT_LOCK = threading.Lock()


def _kobbq_steer_rows(result):
    if result is None:
        return []

    def mark(pred_idx, target):
        if target is None:
            return "-"
        return "✓" if pred_idx == target else "✗"

    rows = []
    for cond in ("base", "alphasteer"):
        r = result[cond]
        rows.append(
            {
                "cond": cond,
                "letter": "ABC"[r["pred_idx"]],
                "pred_text": r["pred_text"],
                "probs": r["probs"],
                "answer_mark": mark(r["pred_idx"], result.get("answer_idx")),
                "biased_mark": mark(r["pred_idx"], result.get("biased_idx")),
            }
        )
    return rows


def index(request):
    return render(request, "index.html")


def select_data_type(request):
    return render(request, "selection_menu/select_data_type.html")


@csrf_exempt
def check_custom_data_metric(request):
    if request.method == "POST" and "file" in request.FILES:
        uploaded_file = request.FILES["file"]
        save_dir = os.environ["PYTHONPATH"] + "/MAF/data/custom_data"
        os.makedirs(save_dir, exist_ok=True)

        file_path = save_dir + "/custom.csv"
        with open(file_path, "wb+") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        data_metric = get_custom_tabular_data_metric(file_path)
        return render(
            request,
            "custom_data_metric.html",
            {"data_name": "custom data", "data": data_metric},
        )


@csrf_exempt
def select_action_per_datatype(request, data_type):
    if data_type == "tabular":
        return render(request, "selection_menu/select_tabular_data.html")
    if data_type == "image":
        return render(request, "selection_menu/select_image_data.html")
    if data_type == "text":
        return render(request, "selection_menu/select_text_algorithm.html")
    if data_type == "audio":
        return render(request, "selection_menu/select_audio_algorithm.html")


@csrf_exempt
def load_metric(request, data_type, data_name):
    if data_name in ("credit_union", "movielens"):
        return redirect(f"/app/algorithm/tabular/{data_name}/")
    if data_name in ("four_findings", "imagenet"):
        return redirect(f"/app/algorithm/image/{data_name}/")
    return render(
        request,
        "metrics_loading.html",
        {"data_type": data_type, "data_name": data_name},
    )


@csrf_exempt
def get_data_metric(request, data_type, data_name):
    if data_name in ("four_findings", "imagenet"):
        return redirect(f"/app/algorithm/image/{data_name}/")
    if data_name in ("credit_union", "movielens"):
        return redirect(f"/app/algorithm/tabular/{data_name}/")
    if data_type == "tabular":
        return render(request, "metrics.html", get_tabular_data_metric(data_name))
    if data_type == "image":
        return render(request, "metrics.html", get_image_data_metric(data_name))


@csrf_exempt
def select_algorithm(request, data_type, data_name):
    """
    select algorithm for tabular&image
    """
    if data_type == "tabular":
        return render(
            request,
            "selection_menu/select_tabular_algorithm.html",
            {"data_name": data_name, "data_type": data_type},
        )
    if data_type == "image":
        return render(
            request,
            "selection_menu/select_image_algorithm.html",
            {"data_name": data_name, "data_type": data_type},
        )


@csrf_exempt
def load_mitigation(request, data_name, algorithm_name):
    if algorithm_name == "cpt":
        # 알고리즘 선택 → 인덱스 입력 페이지 → (sample_idx 와 함께 다시 POST) → 로딩
        form_ctx = {"result_url": request.path, "default_idx": DEFAULT_VISION_SAMPLE}
        if "sample_idx" not in request.POST:
            return render(request, "algorithm/get_data_cpt_vision.html", form_ctx)
        try:
            request.session["vision_sample_idx"] = _parse_vision_index(
                request.POST["sample_idx"]
            )
        except ValueError as e:
            return render(
                request,
                "algorithm/get_data_cpt_vision.html",
                dict(form_ctx, error=str(e)),
            )
    return render(
        request, "mitigation_loading.html", {"algorithm_name": algorithm_name}
    )


_DEEMC_ARMS = ("glocal_lambda0", "dee_mc")  # 화면에는 λ_dee 0 vs 0.5 두 열만(U4)
_DEEMC_LABELS = ("Original (λ_dee = 0)", "Mitigated (λ_dee = 0.5)")
# compare_metric.html 과 같은 행 구성. rule: high(높을수록), low(낮을수록), abs0(0에 가까울수록), near(목표값에 가까울수록)
_DEEMC_TABLES = (
    ("A. Recommendation metrics", (
        ("Test RMSE", "test_rmse", "low", None, None),
        ("DEE", "dee", "low", "0.0", None),
    )),
    ("B. Performance metrics", (
        ("Recall", "recall", "high", None, ("pm", "Recall")),
        ("True negative rate", "true_negative_rate", "high", None, ("pm", "TNR")),
        ("False positive rate", "false_positive_rate", "low", None, ("pm", "FPR")),
        ("False negative rate", "false_negative_rate", "low", None, ("pm", "FNR")),
        ("Precision", "precision", "high", None, ("pm", "precision")),
        ("Negative predictive value", "negative_predictive_value", "high", None, ("pm", "NPV")),
        ("False discovery rate", "false_discovery_rate", "low", None, ("pm", "FDR")),
        ("False omission rate", "false_omission_rate", "low", None, ("pm", "FOR")),
        ("Accuracy", "accuracy", "high", None, ("pm", "ACC")),
    )),
    ("C. Classification metrics", (
        ("Error rate", "error_rate", "low", "0", ("bm", "Error Rate")),
        ("Average odds difference", "average_odds_difference", "abs0", "0.0", ("bm", "Average odds difference")),
        ("Average abs odds difference", "average_abs_odds_difference", "low", "0.0", ("bm", "Average abs odds difference")),
        ("Selection rate", "selection_rate", ("near", 0.5), "0.5", ("bm", "Selection rate")),
        ("Disparate impact", "disparate_impact", ("near", 1.0), "1.0", ("bm", "Disparate impact")),
        ("Statistical parity difference", "statistical_parity_difference", "abs0", "0.0", ("bm", "Statistical parity difference")),
        ("Generalized entropy index", "generalized_entropy_index", "low", "0.0", ("bm", "Generalized entropy index")),
        ("Theil index", "theil_index", "low", "0.0", ("bm", "Theil index")),
        ("Equal opportunity difference", "equal_opportunity_difference", "abs0", "0.0", ("bm", "Equal opportunity difference")),
    )),
)  # fmt: skip
_DEEMC_CHART_PERF = ("true_negative_rate", "false_positive_rate", "false_negative_rate", "recall",
                     "precision", "negative_predictive_value", "false_discovery_rate",
                     "false_omission_rate", "accuracy")  # fmt: skip
_DEEMC_CHART_BIAS = ("average_odds_difference", "selection_rate", "disparate_impact",
                     "statistical_parity_difference", "theil_index", "equal_opportunity_difference")  # fmt: skip


def _deemc_values(arm):
    """compare_metric.html 이 쓰는 키로 평탄화(recall=TPR, NPV=1−FOR). None 은 그대로."""
    maf = arm["maf"]
    values = {"test_rmse": arm["test_rmse"], "dee": arm["dee"], **maf}
    values["recall"] = maf["true_positive_rate"]
    fo = maf["false_omission_rate"]
    values["negative_predictive_value"] = None if fo is None else 1.0 - fo
    return {k: (None if v is None else round(float(v), 4)) for k, v in values.items()}


def _deemc_score(rule, value):
    """작을수록 좋은 점수로 변환."""
    if isinstance(rule, tuple):
        return abs(value - rule[1])
    if rule == "high":
        return -value
    if rule == "abs0":
        return abs(value)
    return value


def _deemc_bold(rule, orig, miti):
    if orig is None or miti is None:
        return (False, False)
    a, b = _deemc_score(rule, orig), _deemc_score(rule, miti)
    return (a <= b, b <= a)


def _deemc_context(result):
    orig, miti = (_deemc_values(result["arms"][a]) for a in _DEEMC_ARMS)
    tables = []
    for title, specs in _DEEMC_TABLES:
        rows = []
        for label, key, rule, ideal, tip in specs:
            bold = _deemc_bold(rule, orig[key], miti[key])
            rows.append(
                {
                    "label": label,
                    "orig": orig[key],
                    "miti": miti[key],
                    "orig_bold": bold[0],
                    "miti_bold": bold[1],
                    "ideal": ideal,
                    "tip": tip,
                }
            )
        tables.append(
            {"title": title, "rows": rows, "has_ideal": title.startswith("C.")}
        )
    chart = {
        "performance": {
            "original": [orig[k] for k in _DEEMC_CHART_PERF],
            "mitigated": [miti[k] for k in _DEEMC_CHART_PERF],
        },
        "bias": {
            "original": [orig[k] for k in _DEEMC_CHART_BIAS],
            "mitigated": [miti[k] for k in _DEEMC_CHART_BIAS],
        },
    }
    return {"tables": tables, "chart_json": json.dumps(chart)}  # None → null


@csrf_exempt
def get_mitigation_result(request, data_name: str, algorithm_name: str):
    if algorithm_name == "cpt":
        idx = request.session.get("vision_sample_idx", DEFAULT_VISION_SAMPLE)
        prediction, error = None, None
        try:
            prediction = predict_vision_cpt(idx)
        except ValueError as e:
            error = str(e)
        except (OSError, RuntimeError):
            logging.getLogger(__name__).exception("cpt predict_vision_cpt failed")
            error = "비전판 예측에 실패했습니다. ImageNet·모델 파일과 서버 로그를 확인해 주세요."
        cached = _cpt_vision_cached(idx) if prediction else None
        job = _VISION_JOBS.get(idx)
        status = "idle"
        if prediction and cached is None:
            if not prediction["correct"]:
                status = "excluded"
            elif job is None:
                orphan = _find_trace_pid(idx)
                if orphan:
                    _adopt_vision_trace(idx, orphan)
                else:
                    _start_vision_trace(idx)
                job, status = _VISION_JOBS.get(idx), "running"
            elif job.get("error"):
                status = "failed"
            else:
                status = "running"
        now = datetime.datetime.now(_KST)
        elapsed_min = None
        if job and job.get("started"):
            try:
                started = datetime.datetime.strptime(
                    job["started"], "%Y-%m-%d %H:%M"
                ).replace(tzinfo=_KST)
                elapsed_min = max(0, int((now - started).total_seconds() // 60))
            except ValueError:
                elapsed_min = None
        return render(
            request,
            "algorithm/cpt.html",
            {
                "algorithm_name": algorithm_name,
                "data_name": data_name,
                "metric_keys": _CPT_KEYS,
                "language": None,
                "vision_index": idx,
                "elapsed_min": elapsed_min,
                "prediction": prediction,
                "vision": cached,
                "status": status,
                "job": job,
                "error": error,
            },
        )

    if algorithm_name == "deemc":
        # λ_dee=0 vs 0.5 두 arm 만 표시(U4). 요청마다 라이브 평가, 겹치면 demo 모듈 락으로 직렬화
        result, error = None, None
        try:
            result = compare_lambda(device="auto")
        except Lambda0Missing as e:  # 상대경로만 담긴 안내문 → 그대로 표시
            error = str(e)
        except (
            Exception
        ):  # 그 외(절대경로·torch 내부 문구 포함)는 로그에만, 화면은 일반 문구
            logging.getLogger(__name__).exception("deemc compare_lambda failed")
            error = (
                "DEE-MC 평가에 실패했습니다. 체크포인트(model/deemc, model/deemc/lambda0)·"
                "데이터(data/deemc)와 서버 로그를 확인해 주세요."
            )
        context = {
            "algorithm_name": "DEE-MC",
            "data_name": "MovieLens-1M",
            "protected_attribute": "sex",
            "labels": _DEEMC_LABELS,
            "error": error,
            "tables": [],
            "chart_json": None,
        }
        if result:
            context.update(_deemc_context(result))
        return render(request, "algorithm/deemc.html", context)

    if algorithm_name == "intapt":
        result = load_audio_algorithm(algorithm_name)
        return render(
            request,
            f"algorithm/{algorithm_name}.html",
            {
                "algorithm_name": ALGORITHM_ID2NAME[algorithm_name],
                "test_avg_perf": result["test_avg_perf"],
                "perf_diff": result["perf_diff"],
            },
        )

    if algorithm_name == "fairasr":
        result = load_audio_algorithm(algorithm_name)
        return render(
            request,
            f"algorithm/{algorithm_name}.html",
            {
                "algorithm_name": ALGORITHM_ID2NAME[algorithm_name],
                "result": result,
            },
        )

    if algorithm_name == "sipmlfr":
        result = load_algorithm(data_name, algorithm_name).run(run_five=0)
        result = {k: round(result[k][0], 5) for k in result}
        result_pairs = list(zip(result.Ks, result.best_recalls))
        return render(
            request,
            f"algorithm/{algorithm_name}.html",
            {
                "algorithm_name": ALGORITHM_ID2NAME[algorithm_name],
                "result": result_pairs,
            },
        )

    if algorithm_name == "dmlbg":
        result = inprocessing.dmlbg.migitage_dmlbg()
        html_path = f"algorithm/{algorithm_name}.html"

        if isinstance(result.get("best_recalls"), np.ndarray):
            result["best_recalls"] = result["best_recalls"].tolist()

        Ks = result.get("Ks", [1, 2, 4, 8, 16, 32])  # fallback
        result["recall_pairs"] = list(zip(Ks, result["best_recalls"]))
        return render(
            request,
            html_path,
            {
                "algorithm_name": algorithm_name,
                "data_name": data_name,
                "result": result,
            },
        )

    if algorithm_name == "alincredit":
        model = load_algorithm(data_name, algorithm_name)
        original_metrics_result, miti_result = model.run()
        return render(
            request,
            f"algorithm/{algorithm_name}.html",
            {
                "data_name": data_name,
                "algorithm_name": ALGORITHM_ID2NAME[algorithm_name],
                "protected_attribute": "sex",
                "original": original_metrics_result,
                "mitigated": miti_result,
                "summary": model.summary,
            },
        )

    if algorithm_name == "sbr":
        model = load_algorithm(data_name, algorithm_name)
        original_metrics_result, miti_result = model.run()
        kpi = model.kpi
        summary = {
            "acc_fair": round(kpi["acc_fair"] * 100, 2),
            "acc_nofair": round(kpi["acc_nofair"] * 100, 2),
            "deo": round(kpi["deo_fair_pct"], 2),
            "policy_drop": round(kpi["policy_drop_pct"], 2),
        }
        return render(
            request,
            "algorithm/alinmed.html",
            {
                "data_name": data_name,
                "algorithm_name": ALGORITHM_ID2NAME[algorithm_name],
                "protected_attribute": "sex",
                "original": original_metrics_result,
                "mitigated": miti_result,
                "summary": summary,
            },
        )

    original_metrics_result, miti_result = load_algorithm(
        data_name, algorithm_name
    ).run()

    html_path = "compare_metric.html"
    if algorithm_name in ["slide", "ftm"]:
        html_path = "algorithm/compare_metric_acc_dp.html"
    if algorithm_name in ["fdf"]:
        html_path = "algorithm/compare_metric_acc.html"
    return render(
        request,
        html_path,
        {
            "data_name": data_name,
            "algorithm_name": ALGORITHM_ID2NAME[algorithm_name],
            "protected_attribute": "sex",
            "original": original_metrics_result,
            "mitigated": miti_result,
        },
    )


@csrf_exempt
def get_textdata(request, algorithm_name: str):
    import json

    html_path = f"algorithm/get_data_{algorithm_name}.html"
    result_url = f"/app/text/mitigation/{algorithm_name}/loading/"

    if algorithm_name == "bbg":
        from pathlib import Path

        tem_dir = os.environ["PYTHONPATH"] + "/MAF/data/bbg/KoBBG_templates.csv"
        df_templates = pd.read_csv(tem_dir)

        grouped_templates = {}
        for _, row in df_templates.iterrows():
            cat = row["Category"]
            stere = row["Stereotype"]
            tid = str(row["Template_ID"])

            grouped_templates.setdefault(cat, {}).setdefault(stere, {}).setdefault(
                tid, []
            ).append(
                {
                    "Ambiguous_context": row["Ambiguous_context"],
                    "Disambiguating_context": row["Disambiguating_context"],
                    "Biased_question": row["Biased_question"],
                    "Counter_biased_question": row["Counter-biased_question"],
                    "Biased_question_answer": row["Biased_question_answer"],
                    "Counter_biased_question_answer": row[
                        "Counter-biased_question_answer"
                    ],
                }
            )

        categories = list(grouped_templates.keys())

        return render(
            request,
            html_path,
            {
                "algorithm": algorithm_name,
                "result_url": result_url,
                "categories": categories,
                "grouped_templates_json": json.dumps(grouped_templates),
                "result_url": result_url,
            },
        )

    return render(
        request, html_path, {"algorithm": algorithm_name, "result_url": result_url}
    )


@csrf_exempt
def processing_text_algorithms(request, algorithm_name: str):
    redirect_path = f"text/mitigation/{algorithm_name}/getresult/"
    if algorithm_name in ["concse", "ember"]:
        return render(request, "text_mitigation_loading.html")

    if algorithm_name == "cpt":
        if request.method == "POST":
            try:
                request.session["cpt_sample_indices"] = _parse_cpt_indices(
                    request.POST.get("sample_indices")
                )
            except ValueError as e:
                return render(
                    request,
                    "algorithm/cpt.html",
                    {"error": str(e), "metric_keys": _CPT_KEYS},
                )
            return render(request, "text_mitigation_loading.html")

    if algorithm_name == "rh":
        if request.method == "POST":
            request.session["question"] = request.POST.get("question")
            request.session["prompt_no"] = request.POST.get("prompt_no")
            return render(request, "text_mitigation_loading.html")

    if algorithm_name == "latte":
        if request.method == "POST":
            request.session["sentence"] = request.POST.get("sentence")
            return render(request, "text_mitigation_loading.html")

    if algorithm_name == "crehate":
        if request.method == "POST":
            request.session["context"] = request.POST.get("context")
            request.session["simple"] = request.POST.get("simple")
            request.session["persona"] = request.POST.get("persona")
            return render(request, "text_mitigation_loading.html")

    if algorithm_name == "kobbq":
        if request.method == "POST":
            request.session["prompt_id"] = int(request.POST.get("prompt_id"))
            request.session["model"] = request.POST.get("model")
            request.session["context"] = request.POST.get("context")
            request.session["question"] = request.POST.get("question")
            request.session["choices"] = request.POST.get("choices").split(", ")
            request.session["biased_answer"] = request.POST.get("biased_answer")
            request.session["answer"] = request.POST.get("answer")
            return render(request, "text_mitigation_loading.html")

    if algorithm_name == "kobbqsteer":
        if request.method == "POST":
            choices = [c.strip() for c in request.POST.get("choices", "").split(",")]
            context, question = (
                request.POST.get("context", ""),
                request.POST.get("question", ""),
            )
            # 세션(DB)에 쓰기 전에 자른다; 같은 상한을 compare_sample 도 다시 검사한다
            if (
                len(context) > MAX_CHARS["context"]
                or len(question) > MAX_CHARS["question"]
                or any(len(c) > MAX_CHARS["choice"] for c in choices)
            ):
                error = (
                    f"입력이 너무 깁니다. 문맥 {MAX_CHARS['context']}자, "
                    f"질문 {MAX_CHARS['question']}자, 선택지 {MAX_CHARS['choice']}자 이내로 줄여 주세요."
                )
                return render(request, "algorithm/kobbqsteer.html", {"error": error})
            request.session["context"] = context
            request.session["question"] = question
            request.session["choices"] = [c for c in choices if c]
            request.session["answer"] = request.POST.get("answer") or None
            request.session["biased_answer"] = request.POST.get("biased_answer") or None
            return render(request, "text_mitigation_loading.html")

    if algorithm_name == "bbg":
        if request.method == "POST":
            request.session["category"] = request.POST.get("category")
            request.session["stereotype"] = request.POST.get("stereotype")
            request.session["template_id"] = request.POST.get("template_id")
            return render(request, "text_mitigation_loading.html")


@csrf_exempt
def show_text_algorithm_result(request, algorithm_name: str):
    html_path = f"algorithm/{algorithm_name}.html"
    if algorithm_name == "concse":
        result = inprocessing.concse.mitigate_concse(base_model="mbert_uncased")
        return render(
            request,
            html_path,
            {"algorithm_name": algorithm_name, "spearman_corr": result},
        )

    if algorithm_name == "cpt":
        language, error = None, None
        indices = request.session.get("cpt_sample_indices") or list(DEFAULT_CPT_SAMPLES)
        try:
            with _CPT_LOCK:
                language = mitigate_cpt(indices)
        except ValueError as e:
            error = str(e)
        except (OSError, RuntimeError):
            logging.getLogger(__name__).exception("cpt mitigate_cpt failed")
            error = "언어판 실행에 실패했습니다. 모델·데이터 파일과 서버 로그를 확인해 주세요."
        return render(
            request,
            html_path,
            {
                "algorithm_name": algorithm_name,
                "error": error,
                "metric_keys": _CPT_KEYS,
                "language": language,
                "vision": None,
                "requested": indices,
                "skipped": sorted(set(indices) - set(language["samples"]))
                if language
                else [],
            },
        )

    if algorithm_name == "ember":
        if_result = postprocessing.ember.run_ifeval.migitage_ember_if()
        qa_result = postprocessing.ember.run_qaeval.migitage_ember_qa()

        return render(
            request,
            html_path,
            {
                "algorithm_name": algorithm_name,
                "em_pairs": if_result["em_pairs"],
                "surface_acc": if_result["surface_acc"],
                "absolute_acc": if_result["absolute_acc"],
                "qa_result_acc": qa_result["acc"],
                "qa_result_matrix": qa_result["matrix"],
            },
        )

    if algorithm_name == "rh":
        question = request.session.get("question")
        prompt_no = request.session.get("prompt_no")
        rhm = preprocessing.RepresentativenessHeuristicMitigator()
        result = rhm.run(question, prompt_no)
        if result:
            return render(
                request,
                html_path,
                {
                    "algorithm_name": "Representativeness_Heuristic",
                    "question": question,
                    "prompt_for_mitigation": result["prompt_fair"],
                    "baseline_output": result["baseline"],
                    "mitigation_output": result["mitigation_output"],
                },
            )

    if algorithm_name == "latte":
        sentence = request.session.get("sentence")
        result = check_toxicity(sentence)
        return render(
            request,
            html_path,
            {"sentence": result["sentence"], "toxicity": result["toxicity"]},
        )

    if algorithm_name == "crehate":
        context = request.session.get("context")
        simple = request.session.get("simple")
        persona = request.session.get("persona")
        result = check_hatespeech(
            context=context, simple=(simple == "True"), persona=(persona == "True")
        )
        return render(
            request,
            html_path,
            {
                "context": context,
                "simple": simple,
                "persona": persona,
                "result": result,
            },
        )

    if algorithm_name == "kobbq":
        prompt_id = request.session.get("prompt_id")
        model = request.session.get("model")
        context = request.session.get("context")
        question = request.session.get("question")
        choices = request.session.get("choices")
        biased_answer = request.session.get("biased_answer")
        answer = request.session.get("answer")
        args = KoBBQArguments(
            prompt_id=prompt_id,
            context=context,
            question=question,
            choices=choices,
            biased_answer=biased_answer,
            answer=answer,
        )
        result = check_korean_bias(model_name=model, data_args=args)
        return render(request, html_path, {"result": result})

    if algorithm_name == "kobbqsteer":
        keys = ("context", "question", "choices", "answer", "biased_answer")
        kwargs = {k: request.session.get(k) for k in keys}
        result, error = None, None
        if not (kwargs["choices"] and kwargs["context"] and kwargs["question"]):
            error = "입력이 없습니다. 입력 페이지에서 다시 시작해 주세요."
        else:
            try:
                result = compare_sample(**kwargs)
            except ValueError as e:
                error = str(e)
            except (OSError, RuntimeError):
                # 경로·GPU 상태는 로그에만 남기고 화면에는 일반 문구만 보인다
                logging.getLogger(__name__).exception(
                    "kobbqsteer compare_sample failed"
                )
                error = "모델 파일을 찾지 못했거나 실행에 실패했습니다. 서버 로그를 확인해 주세요."
        summary, metric_keys = _kobbq_steer_summary()
        summary_rows = (
            [
                {"cond": c, "values": [summary[c][k] for k in metric_keys]}
                for c in ("base", "alphasteer")
            ]
            if summary is not None
            else []
        )
        return render(
            request,
            html_path,
            {
                "result": result,
                "error": error,
                "inputs": {k: kwargs[k] for k in ("context", "question", "choices")},
                "summary": summary,
                "metric_keys": metric_keys,
                "rows": _kobbq_steer_rows(result),
                "summary_rows": summary_rows,
            },
        )

    if algorithm_name == "bbg":
        category = request.session.get("category")
        stereotype = request.session.get("stereotype")
        template_id = request.session.get("template_id")
        args = BBGArguments(
            category=category,
            stereotype=stereotype,
            template_id=template_id,
        )

        result = run_bbg_pipeline(args)
        return render(request, html_path, {"result": result})
