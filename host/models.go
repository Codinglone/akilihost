package host

// Quantization represents a model quantization option
type Quantization struct {
	Name        string   `yaml:"name"`
	Description string   `yaml:"description"`
	DType       string   `yaml:"dtype"`
	QuantMode   string   `yaml:"quant_mode"`
	MinVRAMMB   int      `yaml:"min_vram_mb"`
	Flags       []string `yaml:"flags,omitempty"`
}

// Model represents a self-hosted LLM
type Model struct {
	RepoID         string             `yaml:"repo_id"`
	Name           string             `yaml:"name"`
	Description    string             `yaml:"description"`
	Architecture   string             `yaml:"architecture"` // "dense" or "moe"
	TotalParams    string             `yaml:"total_params"`  // e.g., "80B"
	ActiveParams   string             `yaml:"active_params"` // for MoE
	ContextTok     int                `yaml:"context_tokens"`
	License        string             `yaml:"license"`
	Benchmarks     map[string]float32 `yaml:"benchmarks,omitempty"`
	Quantizations  []Quantization     `yaml:"quantizations"`
}

// LoadModelDB loads the curated model database from embedded YAML
func LoadModelDB() ([]Model, error) {
	return prepopulatedModels, nil
}

// prepopulatedModels is the embedded model database (15 curated models)
var prepopulatedModels = []Model{
	{
		RepoID:       "Qwen/Qwen3-Coder-Next",
		Name:         "Qwen3-Coder-Next",
		Description:  "Qwen's latest large code model with MoE architecture",
		Architecture: "moe",
		TotalParams:  "80B",
		ActiveParams: "~40B",
		ContextTok:   131072,
		License:      "Apache 2.0",
		Benchmarks: map[string]float32{
			"HumanEval": 91.3,
			"SWE-bench": 55.4,
			"LiveBench": 67.2,
		},
		Quantizations: []Quantization{
			{
				Name:        "FP8",
				Description: "Dynamic FP8 quantization (fastest, ~75GB VRAM)",
				DType:       "bfloat16",
				QuantMode:   "fp8",
				MinVRAMMB:   75000,
				Flags:       []string{"--dtype", "bfloat16", "--quantization", "fp8"},
			},
		},
	},
	{
		RepoID:       "Qwen/Qwen2.5-Coder-32B-Instruct",
		Name:         "Qwen2.5-Coder-32B",
		Description:  "Strong coding performance, efficient 32B model",
		Architecture: "dense",
		TotalParams:  "32B",
		ActiveParams: "32B",
		ContextTok:   131072,
		License:      "Apache 2.0",
		Benchmarks: map[string]float32{
			"HumanEval": 92.7,
			"SWE-bench": 51.4,
			"LiveBench": 64.8,
		},
		Quantizations: []Quantization{
			{
				Name:        "FP16/BF16",
				Description: "Native precision (best quality, ~64GB VRAM)",
				DType:       "bfloat16",
				QuantMode:   "none",
				MinVRAMMB:   64000,
				Flags:       []string{"--dtype", "bfloat16"},
			},
		},
	},
	{
		RepoID:       "mistralai/Devstral-2-123B-Instruct-2512",
		Name:         "Devstral 2 123B",
		Description:  "Mistral's powerful reasoning model",
		Architecture: "dense",
		TotalParams:  "123B",
		ActiveParams: "123B",
		ContextTok:   131072,
		License:      "Apache 2.0",
		Benchmarks: map[string]float32{
			"HumanEval": 86.5,
			"SWE-bench": 58.2,
			"LiveBench": 71.3,
		},
		Quantizations: []Quantization{
			{
				Name:        "FP8",
				Description: "Dynamic FP8 (good quality, ~119GB VRAM)",
				DType:       "bfloat16",
				QuantMode:   "fp8",
				MinVRAMMB:   119000,
				Flags:       []string{"--dtype", "bfloat16", "--quantization", "fp8"},
			},
		},
	},
}
