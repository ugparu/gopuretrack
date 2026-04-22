package puretrack

import (
	"math"
	"testing"

	"github.com/ugparu/gopuretrack/embedding"
)

// synthDet is a small Detection implementation used by unit tests. It
// satisfies every optional sub-interface so a single type can exercise the
// full range of tracker behaviors.
type synthDet struct {
	xyxy  [4]float64
	score float64
	cls   int
	idx   int
	emb   []float64
}

func (d *synthDet) GetXYXY() [4]float64     { return d.xyxy }
func (d *synthDet) GetScore() float64       { return d.score }
func (d *synthDet) GetClass() int           { return d.cls }
func (d *synthDet) GetDetID() int           { return d.idx }
func (d *synthDet) GetEmbedding() []float64 { return d.emb }

var (
	_ Detection              = (*synthDet)(nil)
	_ DetectionWithEmbedding = (*synthDet)(nil)
	_ DetectionWithClass     = (*synthDet)(nil)
	_ DetectionWithIndex     = (*synthDet)(nil)
)

// baseTestConfig returns a fully-populated default Config. Tests that want
// to vary a single knob copy this and override the fields they care about,
// so no test implicitly depends on whatever BaseConfig happens to be.
func baseTestConfig() Config {
	return Config{
		TrackNewThresh:         0.6,
		DetHighThresh:          0.6,
		DetLowThresh:           0.1,
		MatchDetHighThresh:     0.9,
		MatchDetLowThresh:      0.5,
		UnconfMatchThresh:      0.7,
		RemoveDuplicateThresh:  0.15,
		MaxFramesLost:          36,
		MaxFramesReidable:      150,
		IoUEmbThresh:           0.38,
		EmbIoUThresh:           0.38,
		EmbThresh:              0.5,
		VRThresh:               0.3,
		EmbReIDThresh:          0.15,
		IoUReIDAlpha:           0.9,
		WithReID:               false,
		WithEmbReactivation:    true,
		MinPureCnt:             15,
		EmbMode:                embedding.ModeEMA,
		EmbEMAAlpha:            0.9,
		KFStdWeightProcessPos:  0.05,
		KFStdWeightProcessVel:  0.00625,
		KFStdWeightMeasurement: 0.05,
		KFInitPosCovFactor:     2.0,
		KFInitVelCovFactor:     10.0,
	}
}

// TestConfigFillDefaultsFromBase: a caller that only sets a few fields must
// still get a fully populated tracker, while the caller-provided values
// have to round-trip unchanged.
func TestConfigFillDefaultsFromBase(t *testing.T) {
	cfg := Config{
		DetHighThresh: 0.6,
		DetLowThresh:  0.1,
	}
	tr := New[*synthDet](cfg)

	got := tr.Config
	if got.MaxFramesLost != BaseConfig.MaxFramesLost {
		t.Errorf("MaxFramesLost: got %d, want %d (from BaseConfig)",
			got.MaxFramesLost, BaseConfig.MaxFramesLost)
	}
	if got.IoUReIDAlpha != BaseConfig.IoUReIDAlpha {
		t.Errorf("IoUReIDAlpha: got %v, want %v", got.IoUReIDAlpha, BaseConfig.IoUReIDAlpha)
	}
	if got.KFStdWeightProcessPos != BaseConfig.KFStdWeightProcessPos {
		t.Errorf("KFStdWeightProcessPos: got %v, want %v",
			got.KFStdWeightProcessPos, BaseConfig.KFStdWeightProcessPos)
	}
	if got.DetHighThresh != 0.6 || got.DetLowThresh != 0.1 {
		t.Errorf("caller-set thresholds lost: got high=%v low=%v",
			got.DetHighThresh, got.DetLowThresh)
	}
}

// TestConfigFillDefaultsLeavesBoolsAlone: fillDefaults must not override
// bool fields, because zero is a meaningful value for them. A caller who
// explicitly sets WithEmbReactivation to false must see it honored even
// when the default is true.
func TestConfigFillDefaultsLeavesBoolsAlone(t *testing.T) {
	cfg := baseTestConfig()
	cfg.WithEmbReactivation = false

	tr := New[*synthDet](cfg)
	if tr.WithEmbReactivation {
		t.Fatal("WithEmbReactivation was flipped back to true by fillDefaults")
	}
}

func TestSetImageSize(t *testing.T) {
	tr := New[*synthDet](baseTestConfig())
	tr.SetImageSize(1920, 1080)
	if tr.imageW != 1920 || tr.imageH != 1080 {
		t.Fatalf("SetImageSize stored %dx%d, want 1920x1080", tr.imageW, tr.imageH)
	}
}

func TestFrameIDIncrementsEachUpdate(t *testing.T) {
	tr := New[*synthDet](baseTestConfig())
	if tr.FrameID() != 0 {
		t.Fatalf("initial FrameID=%d, want 0", tr.FrameID())
	}
	tr.Update(nil)
	tr.Update(nil)
	tr.Update(nil)
	if got := tr.FrameID(); got != 3 {
		t.Fatalf("FrameID after 3 Update calls=%d, want 3", got)
	}
}

// TestTrackBirthAndConfirm covers the unconfirmed → confirmed transition:
// tracks born on frame 1 are activated immediately, but tracks born later
// must survive a second match to stay alive.
func TestTrackBirthAndConfirm(t *testing.T) {
	tr := New[*synthDet](baseTestConfig())

	// Frame 1: two high-confidence detections → two activated tracks.
	active, _ := tr.Update([]*synthDet{
		{xyxy: [4]float64{100, 100, 150, 200}, score: 0.9, idx: 0},
		{xyxy: [4]float64{400, 200, 460, 320}, score: 0.8, idx: 1},
	})
	if len(active) != 2 {
		t.Fatalf("frame 1 active tracks: got %d, want 2", len(active))
	}

	// Frame 2: only one detection matches the first track.
	active, _ = tr.Update([]*synthDet{
		{xyxy: [4]float64{101, 101, 151, 201}, score: 0.9, idx: 0},
	})
	if len(active) != 1 {
		t.Fatalf("frame 2 active tracks: got %d, want 1", len(active))
	}
	if active[0].GetID() != 1 {
		t.Errorf("frame 2 active track id: got %d, want 1", active[0].GetID())
	}
}

// TestLostToRemovedWithoutReidPool verifies that a Lost track with too few
// pure observations is Removed rather than promoted to Reidable once
// MaxFramesLost elapses.
func TestLostToRemovedWithoutReidPool(t *testing.T) {
	cfg := baseTestConfig()
	cfg.MaxFramesLost = 3
	cfg.WithEmbReactivation = true
	cfg.MinPureCnt = 100 // unreachable, so no track qualifies as Reidable
	tr := New[*synthDet](cfg)

	tr.Update([]*synthDet{
		{xyxy: [4]float64{100, 100, 200, 200}, score: 0.9, idx: 0},
	})
	// Feed five empty frames so the track has time to go Lost on frame 2
	// and then cross the MaxFramesLost threshold; aggregating the removed
	// list means the test does not depend on exactly which frame ages it
	// out.
	var allRemoved []Track[*synthDet]
	for range 5 {
		_, removed := tr.Update(nil)
		allRemoved = append(allRemoved, removed...)
	}
	if len(allRemoved) != 1 {
		t.Fatalf("expected exactly one Removed track over the window, got %d", len(allRemoved))
	}
	if allRemoved[0].GetID() != 1 {
		t.Errorf("removed track id: got %d, want 1", allRemoved[0].GetID())
	}
	if _, still := tr.lost[1]; still {
		t.Error("track 1 still present in lost pool after removal")
	}
}

// TestTrackExtInterface documents that the concrete track type satisfies
// TrackExt[T], so callers who want score / class / detID / state can
// type-assert without a runtime surprise.
func TestTrackExtInterface(t *testing.T) {
	tr := New[*synthDet](baseTestConfig())
	active, _ := tr.Update([]*synthDet{
		{xyxy: [4]float64{0, 0, 50, 50}, score: 0.95, cls: 7, idx: 3},
	})
	if len(active) != 1 {
		t.Fatalf("expected 1 track, got %d", len(active))
	}
	ext, ok := active[0].(TrackExt[*synthDet])
	if !ok {
		t.Fatal("Track[*synthDet] does not satisfy TrackExt[*synthDet]")
	}
	if ext.GetScore() != 0.95 {
		t.Errorf("GetScore=%v want 0.95", ext.GetScore())
	}
	if ext.GetClass() != 7 {
		t.Errorf("GetClass=%v want 7", ext.GetClass())
	}
	if ext.GetDetID() != 3 {
		t.Errorf("GetDetID=%v want 3", ext.GetDetID())
	}
	if ext.GetState() != StateTracked {
		t.Errorf("GetState=%v want StateTracked", ext.GetState())
	}
}

// TestXYXYRoundTripsWithinMeasurementNoise drives one detection through
// track birth and checks that GetXYXY returns the input box byte-for-byte
// (the initialization writes the measurement directly into the state).
func TestXYXYRoundTripsWithinMeasurementNoise(t *testing.T) {
	tr := New[*synthDet](baseTestConfig())
	want := [4]float64{100, 200, 300, 500}
	active, _ := tr.Update([]*synthDet{
		{xyxy: want, score: 0.9, idx: 0},
	})
	if len(active) != 1 {
		t.Fatalf("expected 1 track, got %d", len(active))
	}
	got := active[0].GetXYXY()
	for i := range 4 {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			t.Errorf("coord %d: got %v want %v", i, got[i], want[i])
		}
	}
}
