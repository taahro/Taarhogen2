from scripts.void_probe import GIB, MODEL_VRAM_PLANNING_GB, assess_readiness


def test_cpu_only_runtime_is_not_marked_ready():
    result = assess_readiness({"torchAvailable": True, "cudaAvailable": False, "devices": []})
    assert result["status"] == "cpu_only"


def test_missing_torch_marks_probe_incomplete():
    result = assess_readiness({"torchAvailable": False, "cudaAvailable": False, "devices": []})
    assert result["status"] == "probe_incomplete"


def test_low_total_vram_is_flagged_before_model_download():
    gpu = {
        "torchAvailable": True,
        "cudaAvailable": True,
        "devices": [{"totalVramBytes": 8 * GIB, "freeVramBytes": 7 * GIB}],
    }
    result = assess_readiness(gpu)
    assert result["status"] == "below_planning_vram_estimate"


def test_enough_reported_memory_is_only_a_candidate_for_load_test():
    required = MODEL_VRAM_PLANNING_GB * GIB
    gpu = {
        "torchAvailable": True,
        "cudaAvailable": True,
        "devices": [{"totalVramBytes": required + GIB, "freeVramBytes": required + GIB}],
    }
    result = assess_readiness(gpu)
    assert result["status"] == "candidate_for_load_test"
    assert "not proof" in result["reason"].lower()


def test_total_vram_is_not_enough_when_free_memory_is_low():
    required = MODEL_VRAM_PLANNING_GB * GIB
    gpu = {
        "torchAvailable": True,
        "cudaAvailable": True,
        "devices": [{"totalVramBytes": required + GIB, "freeVramBytes": 2 * GIB}],
    }
    result = assess_readiness(gpu)
    assert result["status"] == "low_free_vram"
