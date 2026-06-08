import logging
import platform
import os

import cv2

log = logging.getLogger("video_anonymizer.hardware")

COMPUTE_MODES = ("auto", "gpu", "cpu")


def detect_cpu_info():
    info = {"cores": os.cpu_count() or 1, "arch": platform.machine(), "system": platform.system()}
    try:
        info["brand"] = platform.processor() or "unknown"
    except Exception:
        info["brand"] = "unknown"
    return info


def detect_cuda():
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        if count > 0:
            props = []
            for i in range(count):
                try:
                    cv2.cuda.setDevice(i)
                    name = cv2.cuda.printCudaDeviceInfo(i)
                    props.append({"index": i, "name": name or f"GPU {i}"})
                except Exception:
                    props.append({"index": i, "name": f"GPU {i}"})
            cv2.cuda.setDevice(0)
            return {"available": True, "device_count": count, "devices": props}
    except Exception:
        pass
    return {"available": False, "device_count": 0, "devices": []}


def detect_opencl():
    try:
        have = cv2.ocl.haveOpenCL()
        if have:
            activated = cv2.ocl.isOpenCLActivated()
            return {"available": True, "activated": activated}
    except Exception:
        pass
    return {"available": False, "activated": False}


def detect_opencv_optimized():
    try:
        return cv2.useOptimized()
    except Exception:
        return False


def detect_hardware():
    cuda = detect_cuda()
    opencl = detect_opencl()
    cpu = detect_cpu_info()
    return {
        "cpu": cpu,
        "cuda": cuda,
        "opencl": opencl,
        "opencv_optimized": detect_opencv_optimized(),
    }


RECOMMENDATION_TEXT = {
    "gpu": (
        "GPU je vhodný pro větší videa a vyšší rozlišení. "
        "Detekce a blur poběží rychleji, ale spotřeba energie je vyšší."
    ),
    "cpu": (
        "CPU režim je univerzální a stabilní. "
        "Vhodný pro krátká videa nebo když GPU není k dispozici. "
        "Využije všechna jádra procesoru pro paralelní zpracování."
    ),
}


def recommend_compute_mode(hw, preference="auto"):
    cuda_ok = hw["cuda"]["available"]
    opencl_ok = hw["opencl"]["available"]

    if preference == "gpu":
        if cuda_ok or opencl_ok:
            return "gpu"
        log.warning("GPU preferred but not detected — falling back to CPU")
        return "cpu"

    if preference == "cpu":
        return "cpu"

    cores = hw["cpu"]["cores"]
    if cuda_ok:
        return "gpu"
    if opencl_ok:
        return "gpu"
    if cores >= 8:
        return "cpu"
    return "cpu"


def apply_compute_mode(compute, log):
    hw = detect_hardware()
    mode = recommend_compute_mode(hw, compute)

    cores = hw["cpu"]["cores"]
    if mode == "gpu":
        try:
            cv2.setUseOptimized(True)
            cv2.setNumThreads(cores * 2)
        except Exception:
            pass
        try:
            cv2.ocl.setUseOpenCL(True)
        except Exception:
            pass
        log.info("Compute mode: GPU  (CUDA=%s, OpenCL=%s, cores=%d)",
                 hw["cuda"]["available"], hw["opencl"]["activated"], cores)
    else:
        try:
            cv2.setUseOptimized(True)
            cv2.setNumThreads(cores)
        except Exception:
            pass
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
        log.info("Compute mode: CPU  (cores=%d, OpenCL disabled)", cores)

    return hw, mode


def format_hardware_report(hw):
    lines = []
    lines.append("== Hardware Report ===============================")
    lines.append(f"  CPU: {hw['cpu']['cores']} jader, {hw['cpu']['arch']}")
    lines.append(f"  CUDA GPU: {'ano' if hw['cuda']['available'] else 'ne'} "
                 f"({hw['cuda']['device_count']} zarizeni)")
    lines.append(f"  OpenCL: {'ano' if hw['opencl']['available'] else 'ne'} "
                 f"({'(aktivovan)' if hw['opencl']['activated'] else '(neaktivni)'})")
    lines.append(f"  OpenCV optimalizace: {'zapnuty' if hw['opencv_optimized'] else 'vypnuty'}")
    lines.append("==================================================")
    return "\n".join(lines)
