package host

import "strings"

// SelectBackend decides which inference backend to use.
// If the model explicitly sets backend to "llama-cpp", use it.
// If the model is "vllm" but the quant's VRAM requirement exceeds
// available VRAM (85% of total), auto-fallback to "llama-cpp"
// (which can split across GPU + CPU RAM).
func SelectBackend(model *Model, quant *Quantization, gpu *GPUInfo) string {
	if model.Backend == "llama-cpp" {
		return "llama-cpp"
	}

	availableVRAM := gpu.TotalVRAMMB * 85 / 100
	if quant.MinVRAMMB > availableVRAM {
		return "llama-cpp"
	}

	return "vllm"
}

// ServiceName generates the systemd service name for a model.
// llama-cpp backend: "akilihost-<ModelName>" (spaces replaced with "-")
// vllm backend: "vllm-<ModelName>" (spaces replaced with "-")
func ServiceName(model *Model) string {
	prefix := "vllm"
	if model.Backend == "llama-cpp" {
		prefix = "akilihost"
	}
	sanitized := strings.ReplaceAll(model.Name, " ", "-")
	return prefix + "-" + sanitized
}
