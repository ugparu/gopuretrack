package puretrack

import (
	"math"
	"testing"

	"github.com/ugparu/gopuretrack/embedding"
)

// ---------------------------------------------------------------------------
// prepareDetections: conf-based split and low-conf emb stripping
// ---------------------------------------------------------------------------

func TestPrepareDetectionsSplitsByThresholds(t *testing.T) {
	tr := New[*synthDet](baseTestConfig())
	dets := []*synthDet{
		{xyxy: [4]float64{0, 0, 10, 10}, score: 0.95, idx: 0}, // high
		{xyxy: [4]float64{0, 0, 10, 10}, score: 0.55, idx: 1}, // low (>0.1, ≤0.6)
		{xyxy: [4]float64{0, 0, 10, 10}, score: 0.05, idx: 2}, // dropped
	}
	high, low := tr.prepareDetections(dets)
	if len(high) != 1 || high[0].detID != 0 {
		t.Fatalf("high split: got %d entries (ids=%v), want 1 (id=0)",
			len(high), detIDs(high))
	}
	if len(low) != 1 || low[0].detID != 1 {
		t.Fatalf("low split: got %d entries (ids=%v), want 1 (id=1)",
			len(low), detIDs(low))
	}
}

// Low-confidence detections must not carry their embeddings into the
// second association pass — a low-score observation is too unreliable a
// signal to feed the appearance EMA, and letting it through poisons the
// track's stored embedding against future high-quality matches.
func TestPrepareDetectionsStripsEmbsFromLowSplit(t *testing.T) {
	tr := New[*synthDet](baseTestConfig())
	emb := []float64{1, 0, 0, 0}
	dets := []*synthDet{
		{xyxy: [4]float64{0, 0, 10, 10}, score: 0.9, idx: 0, emb: emb},
		{xyxy: [4]float64{50, 0, 60, 10}, score: 0.4, idx: 1, emb: emb},
	}
	high, low := tr.prepareDetections(dets)
	if len(high) != 1 || len(low) != 1 {
		t.Fatalf("split: high=%d low=%d", len(high), len(low))
	}
	if high[0].emb == nil {
		t.Error("high-conf emb was stripped (should pass through)")
	}
	if low[0].emb != nil {
		t.Error("low-conf emb was not stripped before second association")
	}
}

// ---------------------------------------------------------------------------
// computePureDetectionIDs: visibility + area gating
// ---------------------------------------------------------------------------

func TestComputePureDetectionIDs(t *testing.T) {
	tr := New[*synthDet](baseTestConfig())
	tr.SetImageSize(100, 100) // tiny to exercise the area path

	// Two well-separated boxes → vis=1 for both. One is large (area ≥ 1e-3 of
	// the 100x100 image), one is tiny (below the area floor).
	detsHigh := []detInfo[*synthDet]{
		{detID: 10, xywh: [4]float64{10, 10, 20, 20}, overlap: 0.0}, // vis=1, area=0.04 → pure
		{detID: 11, xywh: [4]float64{80, 80, 2, 2}, overlap: 0.0},   // vis=1, area=4e-4 → NOT pure
	}
	pure := tr.computePureDetectionIDs(detsHigh)
	if len(pure) != 1 {
		t.Fatalf("pure set size: got %d, want 1 (large box only)", len(pure))
	}
	if _, ok := pure[10]; !ok {
		t.Error("large box id=10 missing from pure set")
	}

	// Occluded box (overlap=0.8 → vis=0.2 ≤ VRThresh=0.3) must be excluded
	// even if area is huge.
	occluded := []detInfo[*synthDet]{
		{detID: 20, xywh: [4]float64{50, 50, 90, 90}, overlap: 0.8},
	}
	pure = tr.computePureDetectionIDs(occluded)
	if len(pure) != 0 {
		t.Errorf("occluded box admitted to pure set: %v", pure)
	}
}

func TestComputePureDetectionIDsSkipsAreaWhenNoImageSize(t *testing.T) {
	tr := New[*synthDet](baseTestConfig())
	// Leave imageW/imageH zero — area filter is disabled, only vr matters.
	detsHigh := []detInfo[*synthDet]{
		{detID: 1, xywh: [4]float64{0, 0, 1, 1}, overlap: 0.0}, // tiny but unoccluded
	}
	pure := tr.computePureDetectionIDs(detsHigh)
	if _, ok := pure[1]; !ok {
		t.Error("tiny unoccluded box excluded when area filter is disabled")
	}
}

// ---------------------------------------------------------------------------
// mergeInto
// ---------------------------------------------------------------------------

// mergeInto must EMA-blend both the regular emb and the pure_emb from the
// source into the destination. Skipping the pure_emb blend causes subsequent
// re-identifications in the same sequence to drift, so this test pins the
// exact blended values.
//
// With conf = -1 (the no-confidence branch), α = 1 - EMAAlpha = 0.1:
//
//	dst' = dst · (1-α) + normalize(src) · α
func TestMergeIntoEMABlendsBothEmbeddings(t *testing.T) {
	tr := New[*synthDet](Config{
		EmbMode:     embedding.ModeEMA,
		EmbEMAAlpha: 0.9,
	})

	dst := &track[*synthDet]{
		id:      1,
		emb:     []float64{1, 0, 0, 0},
		pureEmb: []float64{1, 0, 0, 0},
	}
	src := &track[*synthDet]{
		id:      2,
		emb:     []float64{0, 1, 0, 0},
		pureEmb: []float64{0, 1, 0, 0},
		mean:    [8]float64{10, 20, 30, 40, 0, 0, 0, 0},
		cov:     [64]float64{},
		frameID: 42,
		state:   StateTracked,
		conf:    0.8,
		detID:   7,
	}

	tr.mergeInto(dst, src)

	if dst.mean != src.mean || dst.frameID != 42 || dst.state != StateTracked {
		t.Error("mergeInto did not copy motion/bookkeeping fields")
	}
	if dst.conf != 0.8 || dst.detID != 7 {
		t.Error("mergeInto did not copy detection-side fields (breaks id-stability)")
	}

	// The src embeddings are already unit-length, so the blend is simply
	//   dst' = [1,0,0,0]·0.9 + [0,1,0,0]·0.1 = [0.9, 0.1, 0, 0].
	// The tolerance is loose because the EMA step rounds through float32.
	want := []float64{0.9, 0.1, 0, 0}
	for _, pair := range []struct {
		name string
		got  []float64
	}{
		{"emb", dst.emb},
		{"pureEmb", dst.pureEmb},
	} {
		for i, w := range want {
			if math.Abs(pair.got[i]-w) > 1e-5 {
				t.Errorf("%s[%d] = %v, want %v", pair.name, i, pair.got[i], w)
			}
		}
	}
}

// The whole point of a re-id merge is to keep the older (destination)
// track's identity — id, pureCnt and startFrame must survive untouched.
func TestMergeIntoPreservesDstIdentity(t *testing.T) {
	tr := New[*synthDet](Config{EmbMode: embedding.ModeEMA, EmbEMAAlpha: 0.9})
	dst := &track[*synthDet]{
		id:         5,
		pureCnt:    20,
		startFrame: 3,
		emb:        []float64{1, 0},
		pureEmb:    []float64{1, 0},
	}
	src := &track[*synthDet]{
		id:         42,
		pureCnt:    99,
		startFrame: 100,
		emb:        []float64{0, 1},
		pureEmb:    []float64{0, 1},
	}
	tr.mergeInto(dst, src)
	if dst.id != 5 {
		t.Errorf("dst.id changed to %d, want 5 (merge must not overwrite id)", dst.id)
	}
	if dst.pureCnt != 20 {
		t.Errorf("dst.pureCnt changed to %d, want 20", dst.pureCnt)
	}
	if dst.startFrame != 3 {
		t.Errorf("dst.startFrame changed to %d, want 3", dst.startFrame)
	}
}

// When the source has never held a pureEmb — e.g. a track that was never
// observed unoccluded — the destination's pureEmb must be left alone
// rather than blended against nil.
func TestMergeIntoSkipsNilEmbeddings(t *testing.T) {
	tr := New[*synthDet](Config{EmbMode: embedding.ModeEMA, EmbEMAAlpha: 0.9})
	dst := &track[*synthDet]{
		id:      1,
		emb:     []float64{1, 0, 0},
		pureEmb: []float64{1, 0, 0},
	}
	src := &track[*synthDet]{
		id:  2,
		emb: []float64{0, 1, 0},
		// src.pureEmb == nil
	}
	tr.mergeInto(dst, src)
	want := []float64{1, 0, 0}
	for i := range 3 {
		if math.Abs(dst.pureEmb[i]-want[i]) > 1e-9 {
			t.Errorf("pureEmb mutated despite nil src.pureEmb: %v", dst.pureEmb)
			break
		}
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func detIDs[T Detection](d []detInfo[T]) []int {
	out := make([]int, len(d))
	for i, x := range d {
		out[i] = x.detID
	}
	return out
}
