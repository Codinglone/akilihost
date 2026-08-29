package host

import (
	"path/filepath"
	"strconv"
)

// BuildLlamaServerCommand constructs the llama-server argument list
// for serving a model with split GPU/CPU inference.
func BuildLlamaServerCommand(model *Model, quant *Quantization, modelPath string, port int) []string {
	args := []string{
		"llama-server",
		"--model", modelPath,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--cache-type-k", "q8_0",
		"--cache-type-v", "q8_0",
		"--ctx-size", "32768",
	}
	args = append(args, quant.Flags...)
	return args
}

// ModelDir returns the local directory path for a model's GGUF files.
func ModelDir(model *Model) string {
	return filepath.Join("/opt/akilihost/models", model.Name)
}

// ResolveGGUFFromList filters a list of filenames by the quant's file pattern,
// returning the matching GGUF files (may be multiple for split shards).
func ResolveGGUFFromList(files []string, pattern string) []string {
	var matched []string
	for _, f := range files {
		ok, _ := filepath.Match(pattern, f)
		if ok {
			matched = append(matched, f)
		}
	}
	return matched
}
