package host

import (
	"regexp"
	"strconv"
)

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
	Model        *Model
	Quantization *Quantization
	WeightsMB    int
	KVCacheMB    int
	TotalMB      int
	AvailableMB  int
	HeadroomMB   int
	MaxContext   int
}

var paramRe = regexp.MustCompile(`(\d+(?:\.\d+)?)B`)

func parseBParams(s string) int {
	matches := paramRe.FindStringSubmatch(s)
	if len(matches) < 2 {
		return 0
	}
	f, err := strconv.ParseFloat(matches[1], 64)
	if err != nil {
		return 0
	}
	return int(f * 1000)
}

func (s *ModelSizer) calcKVCache(model *Model, tokens int) int {
	return 2 * model.Layers * model.KVHeads * model.HeadDim * 2 * tokens / (1024 * 1024)
}

func (s *ModelSizer) SizeModel(model *Model, q *Quantization, contextTokens int) *SizingResult {
	totalParams := parseBParams(model.TotalParams)

	weightsMB := int(float64(totalParams) * 1e6 * float64(q.BitsPerWeight) / 8 / (1024 * 1024))

	kvMB := s.calcKVCache(model, contextTokens)

	totalMB := weightsMB + kvMB + 512

	availableMB := s.GPU.TotalVRAMMB * 85 / 100
	headroomMB := availableMB - totalMB

	return &SizingResult{
		Model:        model,
		Quantization: q,
		WeightsMB:    weightsMB,
		KVCacheMB:    kvMB,
		TotalMB:      totalMB,
		AvailableMB:  availableMB,
		HeadroomMB:   headroomMB,
		MaxContext:   contextTokens,
	}
}

func (s *ModelSizer) FindFit(models []Model, availableMB int, contextTokens int) []*SizingResult {
	var results []*SizingResult
	totalMemoryMB := s.GPU.TotalVRAMMB + s.GPU.SystemRAMMB
	for _, model := range models {
		for _, q := range model.Quantizations {
			effectiveLimit := availableMB
			if model.Backend == "llama-cpp" {
				effectiveLimit = totalMemoryMB * 85 / 100
			}

			if q.MinVRAMMB <= effectiveLimit {
				r := s.SizeModel(&model, &q, contextTokens)
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
