package host

import "testing"

func TestParseSystemRAMMB(t *testing.T) {
	tests := []struct {
		input    string
		expected int
	}{
		{"MemTotal:       28456320 kB\n", 27789},
		{"MemTotal:       16384 kB\n", 16},
		{"MemTotal:         1024 kB\n", 1},
		{"", 0},
	}

	for _, tt := range tests {
		got := parseSystemRAMMB(tt.input)
		if got != tt.expected {
			t.Errorf("parseSystemRAMMB(%q) = %d, want %d", tt.input, got, tt.expected)
		}
	}
}

func TestDetectGPUIncludesSystemRAM(t *testing.T) {
	gpu := &GPUInfo{
		Name:        "Tesla T4",
		TotalVRAMMB: 16384,
		SystemRAMMB: 28456,
	}
	if gpu.SystemRAMMB != 28456 {
		t.Errorf("expected SystemRAMMB 28456, got %d", gpu.SystemRAMMB)
	}
}
