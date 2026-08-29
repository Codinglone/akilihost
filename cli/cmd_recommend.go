package cli

import (
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/Codinglone/akilihost/host"
)

var recommendCmd = &cobra.Command{
	Use:   "recommend [model]",
	Short: "Show models that fit your GPU, with benchmarks",
	Args:  cobra.MaximumNArgs(1),
	Run: func(cmd *cobra.Command, args []string) {
		gpu, err := host.DetectGPU()
		if err != nil {
			fmt.Printf("Error detecting GPU: %v\n", err)
			os.Exit(1)
		}

		models, err := host.LoadModelDB()
		if err != nil {
			fmt.Printf("Error loading model DB: %v\n", err)
			os.Exit(1)
		}

		sizer := host.NewModelSizer(gpu)
		maxCtx := 32768
		if cmd.Flags().Changed("max-model-len") {
			maxCtx = serveMaxModelLen
		}
		results := sizer.FindFit(models, gpu.TotalVRAMMB*85/100, maxCtx)

		fmt.Print(renderRecommendation(gpu, results))
	},
}

func init() {
	recommendCmd.Flags().IntVar(&serveMaxModelLen, "max-model-len", 32768, "Max context tokens for sizing estimate")
}

func renderRecommendation(gpu *host.GPUInfo, results []*host.SizingResult) string {
	var buf strings.Builder

	fmt.Fprintf(&buf, "GPU: %s (%d MB VRAM, %d MB RAM)\n\n",
		gpu.Name, gpu.TotalVRAMMB, gpu.SystemRAMMB)

	if len(results) == 0 {
		buf.WriteString("No models fit on this GPU.\n")
		return buf.String()
	}

	fmt.Fprintf(&buf, "%-25s %-12s %-10s %10s %10s\n",
		"Model", "Quant", "Backend", "≈Size", "Headroom")
	buf.WriteString(strings.Repeat("-", 72))
	buf.WriteString("\n")

	for _, r := range results {
		fmt.Fprintf(&buf, "%-25s %-12s %-10s %8dMB %8dMB\n",
			r.Model.Name, r.Quantization.Name, r.Model.Backend,
			r.TotalMB, r.HeadroomMB)
	}

	best := results[0]
	fmt.Fprintf(&buf, "\nRecommended: %s %s (%.1f GB estimated)\n",
		best.Model.Name, best.Quantization.Name,
		float64(best.TotalMB)/1024)

	return buf.String()
}
