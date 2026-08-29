package host

// ModelSizer calculates VRAM requirements and recommends suitable models
type ModelSizer struct {
	GPU *GPUInfo
}

// NewModelSizer creates a new sizer for the given GPU
func NewModelSizer(gpu *GPUInfo) *ModelSizer {
	return &ModelSizer{GPU: gpu}
}

// SizingResult contains VRAM calculations
type SizingResult struct {
	Model          *Model
	Quantization   *Quantization
	WeightsMB      int
	KVCacheMB      int
	TotalMB        int
	AvailableMB    int
	HeadroomMB     int
	MaxContext     int
}

// calcKVCache estimates KV cache memory needs
func (s *ModelSizer) calcKVCache(model *Model, tokens int) int {
	// Rough estimate: ~0.18 MB per token for MoE, ~0.35 MB for dense
	var bytesPerToken float64
	if model.Architecture == "moe" {
		bytesPerToken = 0.18
	} else {
		bytesPerToken = 0.35
	}
	return int(float64(tokens) * bytesPerToken * 1024 * 1024)
}

// SizeModel calculates VRAM requirements for a model + quantization
func (s *ModelSizer) SizeModel(model *Model, q *Quantization) *SizingResult {
	totalParams := model.TotalParams
	var activeParams int = 1
	if model.Architecture == "moe" {
		activeParams = 2 // estimate 2 active experts
	}

	var weightMB int
	if q.QuantMode == "fp8" {
		// FP8: 1 byte per param
		weightMB = parseBParams(totalParams) * activeParams / 8 // /8 for MB
	} else {
		// BF16: 2 bytes per param
		weightMB = parseBParams(totalParams) * 2 * activeParams / 8
	}

	kvMB := s.calcKVCache(model, model.ContextTok)

	totalMB := weightMB + kvMB
	availableMB := s.GPU.TotalVRAMMB * 85 / 100 // use 85% for safety
	headroomMB := availableMB - totalMB

	return &SizingResult{
		Model:         model,
		Quantization:  q,
		WeightsMB:     weightMB,
		KVCacheMB:     kvMB,
		TotalMB:       totalMB,
		AvailableMB:   availableMB,
		HeadroomMB:    headroomMB,
		MaxContext:    model.ContextTok,
	}
}

func parseBParams(s string) int {
	// Parse "80B" -> 80000
	s = s + "B"
	if len(s) < 2 {
		return 0
	}
	// Simplified parsing
	return 80000
}

// FindFit returns the best model + quantization that fits on GPU
func (s *ModelSizer) FindFit(models []Model, availableMB int) []*SizingResult {
	var results []*SizingResult
	totalMemoryMB := s.GPU.TotalVRAMMB + s.GPU.SystemRAMMB
	for _, model := range models {
		for _, q := range model.Quantizations {
			// llama-cpp backend can split across GPU + CPU RAM
			effectiveLimit := availableMB
			if model.Backend == "llama-cpp" {
				effectiveLimit = totalMemoryMB * 85 / 100
			}

			if q.MinVRAMMB <= effectiveLimit {
				r := s.SizeModel(&model, &q)
				r.AvailableMB = effectiveLimit
				r.HeadroomMB = effectiveLimit - q.MinVRAMMB
				if r.HeadroomMB >= 0 {
					results = append(results, r)
				}
			}
		}
	}
	return results
}
