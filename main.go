package main

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"charm.land/bubbles/v2/progress"
	tea "charm.land/bubbletea/v2"
	"charm.land/huh/v2"
	"charm.land/lipgloss/v2"
	"github.com/spf13/pflag"
)

const (
	padding  = 2
	maxWidth = 80
	randSize = 35
)

const charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/~!@#$%^&*_-=.\\"

type preset struct {
	Name     string
	CRF      int
	Speed    string
	Tier     int
	RatioEst float64
}

var presets = []preset{
	{"High Quality (Lower compression)", 22, "medium", 0, 0.70},
	{"Balanced (Recommended)", 26, "slow", 1, 0.45},
	{"Aggressive (Smaller file size)", 32, "slower", 2, 0.25},
}

type encoder struct {
	Name       string
	Codec      string
	Hardware   bool
	Pre        []string
	Args       func(p preset) []string
	SizeFactor float64
}

var softwareEncoder = encoder{
	Name:       "libx264 (CPU)",
	Codec:      "libx264",
	SizeFactor: 1.0,
	Args: func(p preset) []string {
		return []string{
			"-c:v", "libx264",
			"-pix_fmt", "yuv420p",
			"-crf", strconv.Itoa(p.CRF),
			"-preset", p.Speed,
		}
	},
}

var nvenc = encoder{
	Name:       "NVIDIA NVENC (GPU)",
	Codec:      "h264_nvenc",
	Hardware:   true,
	SizeFactor: 1.25,
	Args: func(p preset) []string {
		return []string{
			"-c:v", "h264_nvenc",
			"-pix_fmt", "yuv420p",
			"-preset", fmt.Sprintf("p%d", 5+p.Tier),
			"-tune", "hq",
			"-rc", "vbr",
			"-cq", strconv.Itoa(p.CRF),
			"-b:v", "0",
		}
	},
}

var qsv = encoder{
	Name:       "Intel Quick Sync (GPU)",
	Codec:      "h264_qsv",
	Hardware:   true,
	SizeFactor: 1.25,
	Args: func(p preset) []string {
		return []string{
			"-c:v", "h264_qsv",
			"-pix_fmt", "nv12",
			"-preset", p.Speed,
			"-global_quality", strconv.Itoa(p.CRF),
		}
	},
}

var amf = encoder{
	Name:       "AMD AMF (GPU)",
	Codec:      "h264_amf",
	Hardware:   true,
	SizeFactor: 1.25,
	Args: func(p preset) []string {
		q := strconv.Itoa(p.CRF)
		return []string{
			"-c:v", "h264_amf",
			"-pix_fmt", "yuv420p",
			"-quality", []string{"balanced", "quality", "quality"}[p.Tier],
			"-rc", "cqp",
			"-qp_i", q,
			"-qp_p", q,
		}
	},
}

var videotoolbox = encoder{
	Name:       "Apple VideoToolbox (GPU)",
	Codec:      "h264_videotoolbox",
	Hardware:   true,
	SizeFactor: 1.25,
	Args: func(p preset) []string {
		q := 100 - 3*(p.CRF-12)
		return []string{
			"-c:v", "h264_videotoolbox",
			"-pix_fmt", "yuv420p",
			"-q:v", strconv.Itoa(q),
		}
	},
}

func vaapiEncoder(device string) encoder {
	return encoder{
		Name:       fmt.Sprintf("VAAPI (GPU, %s)", filepath.Base(device)),
		Codec:      "h264_vaapi",
		Hardware:   true,
		SizeFactor: 1.25,
		Pre:        []string{"-vaapi_device", device},
		Args: func(p preset) []string {
			return []string{
				"-vf", "format=nv12,hwupload",
				"-c:v", "h264_vaapi",
				"-qp", strconv.Itoa(p.CRF),
			}
		},
	}
}

func hardwareCandidates() []encoder {
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			return []encoder{videotoolbox}
		}
		return nil
	case "windows":
		return []encoder{nvenc, qsv, amf}
	default:
		list := []encoder{nvenc, qsv}
		devices, _ := filepath.Glob("/dev/dri/renderD*")
		for _, d := range devices {
			list = append(list, vaapiEncoder(d))
		}
		return list
	}
}

func probeEncoder(enc encoder) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	args := []string{"-hide_banner", "-loglevel", "error"}
	args = append(args, enc.Pre...)
	args = append(args, "-f", "lavfi", "-i", "color=c=black:s=320x240:r=30:d=0.5")
	args = append(args, enc.Args(presets[1])...)
	args = append(args, "-f", "null", "-")

	return exec.CommandContext(ctx, "ffmpeg", args...).Run() == nil
}

func detectEncoder() encoder {
	out, err := exec.Command("ffmpeg", "-hide_banner", "-encoders").Output()
	if err != nil {
		return softwareEncoder
	}
	listed := string(out)

	for _, enc := range hardwareCandidates() {
		if strings.Contains(listed, enc.Codec) && probeEncoder(enc) {
			return enc
		}
	}
	return softwareEncoder
}

type model struct {
	program    *tea.Program
	inputPath  string
	outputPath string
	duration   time.Duration
	elapsed    time.Duration
	percent    float64
	phase      string
	err        error
	rand       [randSize]byte
	ctx        context.Context
	cancel     context.CancelFunc
	progress   progress.Model
	preset     preset
	encoder    encoder
	notice     string
	fileSize   int64
	done       bool
}

type progressMsg struct {
	elapsed time.Duration
	percent float64
}

type ffmpegDoneMsg struct{ err error }

type errMsg struct{ err error }

func initialModel(inputPath, outputPath string, dur time.Duration, fileSize int64, pr preset, enc encoder) *model {
	var r [randSize]byte
	copy(r[:], charset)
	return &model{
		inputPath:  inputPath,
		outputPath: outputPath,
		duration:   dur,
		fileSize:   fileSize,
		preset:     pr,
		encoder:    enc,
		phase:      "encoding",
		progress:   progress.New(progress.WithDefaultBlend()),
		rand:       r,
	}
}

type tickMsg time.Time

func tick() tea.Cmd {
	return tea.Tick(time.Second/24, func(t time.Time) tea.Msg {
		return tickMsg(t)
	})
}

func (m *model) Init() tea.Cmd {
	return tea.Batch(m.startFFmpegCmd(), tick())
}

func (m *model) ffmpegArgs() []string {
	args := []string{"-y", "-hide_banner", "-loglevel", "error"}
	args = append(args, m.encoder.Pre...)
	args = append(args, "-i", m.inputPath)
	args = append(args, m.encoder.Args(m.preset)...)
	args = append(args, "-nostats", "-progress", "pipe:1", m.outputPath)
	return args
}

func lastLines(s string, n int) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}

func (m *model) startFFmpegCmd() tea.Cmd {
	args := m.ffmpegArgs()
	ctx, program, total := m.ctx, m.program, m.duration

	return func() tea.Msg {
		cmd := exec.CommandContext(ctx, "ffmpeg", args...)

		var stderr bytes.Buffer
		cmd.Stderr = &stderr

		stdout, err := cmd.StdoutPipe()
		if err != nil {
			return errMsg{err: fmt.Errorf("stdout pipe: %w", err)}
		}

		if err := cmd.Start(); err != nil {
			return errMsg{err: fmt.Errorf("start ffmpeg: %w", err)}
		}

		go func() {
			scanner := bufio.NewScanner(stdout)
			for scanner.Scan() {
				line := scanner.Text()
				val, ok := strings.CutPrefix(line, "out_time_us=")
				if !ok {
					continue
				}
				us, err := strconv.Atoi(val)
				if err != nil || us < 0 {
					continue
				}

				elapsed := time.Duration(us) * time.Microsecond
				percent := min(elapsed.Seconds()/total.Seconds(), 1)
				program.Send(progressMsg{elapsed: elapsed, percent: percent})
			}
			err1 := scanner.Err()
			err2 := cmd.Wait()
			if err2 != nil {
				if detail := lastLines(stderr.String(), 3); detail != "" {
					err2 = fmt.Errorf("%w\n%s", err2, detail)
				}
			}

			program.Send(ffmpegDoneMsg{err: errors.Join(err1, err2)})
		}()

		return nil
	}
}

func (m *model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyPressMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			if m.cancel != nil {
				m.cancel()
			}
			return m, tea.Quit
		}
		return m, nil

	case tea.WindowSizeMsg:
		width := min(msg.Width-padding*2-4, maxWidth)
		m.progress.SetWidth(width)
		return m, nil

	case errMsg:
		m.phase = "error"
		m.err = msg.err
		m.done = true
		return m, tea.Quit

	case progressMsg:
		m.elapsed = msg.elapsed
		m.percent = msg.percent
		cmd := m.progress.SetPercent(m.percent)
		return m, cmd

	case ffmpegDoneMsg:
		if msg.err != nil {
			if m.encoder.Hardware && m.ctx.Err() == nil {
				m.notice = fmt.Sprintf("%s failed on this file, fell back to CPU encoding", m.encoder.Name)
				m.encoder = softwareEncoder
				m.elapsed, m.percent = 0, 0
				return m, tea.Batch(m.progress.SetPercent(0), m.startFFmpegCmd())
			}
			m.done = true
			m.phase = "error"
			m.err = msg.err
			return m, tea.Quit
		}

		m.done = true
		m.phase = "done"
		m.percent = 1.0
		return m, tea.Batch(m.progress.SetPercent(1.0), tea.Quit)

	case progress.FrameMsg:
		var cmd tea.Cmd
		m.progress, cmd = m.progress.Update(msg)
		return m, cmd

	case tickMsg:
		var newRand [randSize]byte
		copy(newRand[:], m.rand[1:])
		newRand[randSize-1] = charset[rand.Intn(len(charset))]
		m.rand = newRand
		return m, tick()

	default:
		return m, nil
	}
}

var (
	helpStyle = lipgloss.NewStyle().Foreground(lipgloss.BrightBlack)
	randStyle = lipgloss.NewStyle().Foreground(lipgloss.Green).Bold(true)
	warnStyle = lipgloss.NewStyle().Foreground(lipgloss.Yellow)
)

func (m *model) View() tea.View {
	buf := bytes.NewBuffer(nil)
	switch m.phase {
	case "ffprobe":
		buf.WriteString("Probing video duration...")
	case "encoding":
		fmt.Fprintf(buf, "Encoding: %s → %s\n", m.inputPath, m.outputPath)
		fmt.Fprintf(buf, "Encoder:  %s\n", m.encoder.Name)
		if m.notice != "" {
			fmt.Fprintln(buf, warnStyle.Render(m.notice))
		}
		buf.WriteByte('\n')
		for range padding {
			buf.WriteByte(' ')
		}
		fmt.Fprintf(buf, "Duration: %.1fs | Elapsed: %.1fs | %.1f%%\n", m.duration.Seconds(), m.elapsed.Seconds(), m.percent*100)
		for range padding {
			buf.WriteByte(' ')
		}
		fmt.Fprintf(buf, "Compressing: [%s]\n\n", randStyle.Render(string(m.rand[:])))
		fmt.Fprintln(buf, m.progress.View())
	case "done":
		fmt.Fprintf(buf, "Encoding complete: %s (%s)", m.outputPath, m.encoder.Name)
		if m.notice != "" {
			fmt.Fprintf(buf, "\n%s", warnStyle.Render(m.notice))
		}
	case "error":
		fmt.Fprintf(buf, "Error: %v", m.err)
	default:
		buf.WriteString("Unknown state")
	}
	buf.WriteByte('\n')
	buf.WriteString(helpStyle.Render("Press q or Ctrl+C to quit"))
	return tea.NewView(buf.String())
}

func probeVideo(path string) (time.Duration, int64, error) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, 0, fmt.Errorf("stat file: %w", err)
	}

	cmd := exec.Command(
		"ffprobe",
		"-v", "error",
		"-show_entries", "format=duration",
		"-of", "default=noprint_wrappers=1:nokey=1",
		path,
	)
	out, err := cmd.Output()
	if err != nil {
		return 0, 0, fmt.Errorf("ffprobe failed: %w", err)
	}

	var sec float64
	if _, err := fmt.Sscanf(string(out), "%f", &sec); err != nil {
		return 0, 0, fmt.Errorf("invalid duration format: %w", err)
	}

	return time.Duration(sec * float64(time.Second)), info.Size(), nil
}

func formatBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}

func main() {
	var cpuOnly bool
	pflag.BoolVarP(&cpuOnly, "cpu", "c", false, "skip GPU detection and encode on the CPU with libx264")
	pflag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [-c,--cpu] <input_video>\n", os.Args[0])
		pflag.PrintDefaults()
	}
	pflag.Parse()

	if pflag.NArg() < 1 {
		pflag.Usage()
		os.Exit(1)
	}
	inputPath := pflag.Arg(0)
	ext := filepath.Ext(inputPath)
	base := strings.TrimSuffix(inputPath, ext)
	outputPath := base + "_compressed" + ext

	fmt.Println("Analyzing video details...")
	dur, size, err := probeVideo(inputPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error probing video: %v\n", err)
		os.Exit(1)
	}

	enc := softwareEncoder
	if !cpuOnly {
		fmt.Println("Detecting hardware encoders...")
		enc = detectEncoder()
	}
	fmt.Printf("Using encoder: %s\n", enc.Name)

	var options []huh.Option[preset]
	for _, p := range presets {
		ratio := min(p.RatioEst*enc.SizeFactor, 0.95)
		estSize := int64(float64(size) * ratio)
		reduction := int(math.Round((1.0 - ratio) * 100))

		label := fmt.Sprintf("%-35s | ~%d%% reduction | Est. Size: %s", p.Name, reduction, formatBytes(estSize))
		options = append(options, huh.NewOption(label, p))
	}

	var selectedPreset preset
	form := huh.NewForm(
		huh.NewGroup(
			huh.NewSelect[preset]().
				Title(fmt.Sprintf("Select Compression Preset (Original: %s, Encoder: %s)", formatBytes(size), enc.Name)).
				Options(options...).
				Value(&selectedPreset),
		),
	)

	if err := form.Run(); err != nil {
		if errors.Is(err, huh.ErrUserAborted) {
			os.Exit(0)
		}
		fmt.Fprintf(os.Stderr, "Prompt error: %v\n", err)
		os.Exit(1)
	}

	m := initialModel(inputPath, outputPath, dur, size, selectedPreset, enc)

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT, syscall.SIGQUIT)
	defer cancel()
	m.ctx = ctx
	m.cancel = cancel

	p := tea.NewProgram(m)
	m.program = p

	if _, err := p.Run(); err != nil {
		fmt.Printf("Alas, there's been an error: %v\n", err)
		os.Exit(1)
	}
}
