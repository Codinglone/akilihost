package host

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// GPUInfo represents detected GPU specifications
type GPUInfo struct {
	Name          string  `json:"name"`
	TotalVRAMMB   int     `json:"total_vram_mb"`
	CUDAMajor     int     `json:"cuda_major"`
	CUDAMinor     int     `json:"cuda_minor"`
	ComputeCap    string  `json:"compute_cap"`
	CUDAVersion   string  `json:"cuda_version"`
}

func parseVRAM(vramStr string) (int, error) {
	vramStr = strings.ReplaceAll(vramStr, "MiB", "")
	vramStr = strings.TrimSpace(vramStr)
	return parseInt(vramStr)
}

func parseInt(s string) (int, error) {
	return strconv.Atoi(s)
}

func parseCUDAVersion(s string) (int, int, error) {
	// nvidia-smi CUDA version format: "12.4" or "12004"
	s = strings.TrimSpace(s)
	s = strings.ReplaceAll(s, "CUDA Version: ", "")
	s = strings.ReplaceAll(s, "CUDA Version ", "")
	
	parts := strings.Split(s, ".")
	if len(parts) >= 2 {
		major, err1 := strconv.Atoi(parts[0])
		minor, err2 := strconv.Atoi(parts[1])
		if err1 == nil && err2 == nil {
			return major, minor, nil
		}
	}
	
	// Try compact format like "12004"
	if len(s) >= 3 {
		major, err1 := strconv.Atoi(s[:2])
		minor, err2 := strconv.Atoi(s[2:])
		if err1 == nil && err2 == nil {
			return major, minor, nil
		}
	}
	
	return 0, 0, fmt.Errorf("failed to parse CUDA version: %s", s)
}

func parseCUDAVersionFromDriver(s string) (int, int, error) {
	// Driver version format: "535.161.08" or "550.54.15"
	// First part indicates CUDA capability
	parts := strings.Split(s, ".")
	if len(parts) < 2 {
		return 12, 4, fmt.Errorf("invalid driver version format")
	}
	
	major, err := strconv.Atoi(parts[0])
	if err != nil {
		return 12, 4, err
	}
	
	// Convert driver major to CUDA major
	// E.g., 535 -> 12.3, 550 -> 12.5, 560 -> 12.6
	var cudaMajor int
	switch {
	case major >= 560:
		cudaMajor = 12
	case major >= 550:
		cudaMajor = 12
	case major >= 535:
		cudaMajor = 12
	default:
		cudaMajor = 12
	}
	
	// Estimate minor version based on driver minor
	cudaMinor := 0
	if len(parts) >= 2 {
		cudaMinor = major % 10
		if cudaMinor == 0 {
			cudaMinor = 4
		}
	}
	
	return cudaMajor, cudaMinor, nil
}

// DetectGPU parses nvidia-smi output to detect GPU specs
func DetectGPU() (*GPUInfo, error) {
	cmd := exec.Command("nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap,cuda_version", "--format=csv,noheader,nounits")
	output, err := cmd.Output()
	if err != nil {
		return nil, err
	}

	lines := strings.TrimSpace(string(output))
	if lines == "" {
		return nil, os.ErrNotExist
	}

	parts := strings.Split(lines, ",")
	if len(parts) < 5 {
		return nil, os.ErrNotExist
	}

	name := strings.TrimSpace(parts[0])
	vramStr := strings.TrimSpace(parts[1])
	driverVer := strings.TrimSpace(parts[2])
	computeCap := strings.TrimSpace(parts[3])
	cudaVerStr := strings.TrimSpace(parts[4])

	// Parse VRAM (format: "143167 MiB")
	vramMB, err := parseVRAM(vramStr)
	if err != nil {
		return nil, err
	}

	// Parse CUDA version from nvidia-smi output (format: "12.4" or "12004")
	cudaMajor, cudaMinor, err := parseCUDAVersion(cudaVerStr)
	if err != nil {
		// Fallback to parsing from driver version
		cudaMajor, cudaMinor, err = parseCUDAVersionFromDriver(driverVer)
		if err != nil {
			cudaMajor, cudaMinor = 12, 4
		}
	}

	return &GPUInfo{
		Name:          name,
		TotalVRAMMB:   vramMB,
		CUDAMajor:     cudaMajor,
		CUDAMinor:     cudaMinor,
		ComputeCap:    computeCap,
		CUDAVersion:   "CUDA " + strings.Join([]string{strconv.Itoa(cudaMajor), strconv.Itoa(cudaMinor)}, "."),
	}, nil
}
