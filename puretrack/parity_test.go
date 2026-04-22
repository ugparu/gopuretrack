package puretrack

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/ugparu/gopuretrack/embedding"
)

// This file is an integration test that replays YOLOX-MOT17 detections
// against the tracker and checks per-frame output (track counts, bounding
// boxes, id stability) against a stored reference.
//
// Fixtures live under testdata/:
//   dets/MOT17-XX.json                detections, one file per sequence
//   embs/MOT17-09.json                MOT17-09 embeddings, capped at the
//                                     first 10 frames to keep the fixture
//                                     small enough to ship in-repo
//   reference/MOT17-XX_no_embs.json   no-ReID reference for all sequences
//   reference/MOT17-09_with_embs.json ReID reference, 10-frame cap
//
// The parity ceiling (bbox ℓ∞ ≈ 7e-5) is bounded by float32 precision in
// the reference pipeline. A diff past ~1e-3 indicates a logic regression,
// not a rounding artifact.

// ---------------------------------------------------------------------------
// Detection type for parity runs
// ---------------------------------------------------------------------------

type parityDet struct {
	xyxy  [4]float64
	score float64
	cls   int
	idx   int
	emb   []float64
}

func (d *parityDet) GetXYXY() [4]float64     { return d.xyxy }
func (d *parityDet) GetScore() float64       { return d.score }
func (d *parityDet) GetClass() int           { return d.cls }
func (d *parityDet) GetDetID() int           { return d.idx }
func (d *parityDet) GetEmbedding() []float64 { return d.emb }

// ---------------------------------------------------------------------------
// Fixture JSON layouts
// ---------------------------------------------------------------------------

// rowsFile mirrors the on-disk layout of testdata/dets/*.json and
// testdata/embs/*.json: a single "rows" key holding the 2-D float64 array
// (upcast from the float32 source).
type rowsFile struct {
	Rows [][]float64 `json:"rows"`
}

type refFrame struct {
	FrameID int         `json:"frame_id"`
	Tracks  [][]float64 `json:"tracks"` // [x1,y1,x2,y2,track_id,conf,cls,det_id]
}

type refFile struct {
	Sequence    string     `json:"sequence"`
	Mode        string     `json:"mode"`
	ImageWidth  int        `json:"image_width"`
	ImageHeight int        `json:"image_height"`
	Frames      []refFrame `json:"frames"`
}

func loadRows(tb testing.TB, path string) [][]float64 {
	tb.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		tb.Fatalf("read %s: %v", path, err)
	}
	var f rowsFile
	if err := json.Unmarshal(data, &f); err != nil {
		tb.Fatalf("parse %s: %v", path, err)
	}
	return f.Rows
}

func loadReference(tb testing.TB, path string) refFile {
	tb.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		tb.Fatalf("read %s: %v", path, err)
	}
	var ref refFile
	if err := json.Unmarshal(data, &ref); err != nil {
		tb.Fatalf("parse %s: %v", path, err)
	}
	return ref
}

// ---------------------------------------------------------------------------
// Tracker factory used by the parity runner
// ---------------------------------------------------------------------------

func newParityTracker(withReID bool) *Tracker[*parityDet] {
	cfg := Config{
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
		WithReID:               withReID,
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
	return New[*parityDet](cfg)
}

// ---------------------------------------------------------------------------
// Per-sequence image sizes (from MOT17 seqinfo.ini)
// ---------------------------------------------------------------------------

var imgSize = map[string]struct{ W, H int }{
	"MOT17-02": {1920, 1080},
	"MOT17-04": {1920, 1080},
	"MOT17-05": {640, 480},
	"MOT17-09": {1920, 1080},
	"MOT17-10": {1920, 1080},
	"MOT17-11": {1920, 1080},
	"MOT17-13": {1920, 1080},
}

// ---------------------------------------------------------------------------
// Parity runner
// ---------------------------------------------------------------------------

type parityResult struct {
	frames        int
	countMismatch int
	matchedDetIDs int
	maxAbs        float64
	idSwitches    int
	refTotal      int
	goTotal       int
}

// runParity drives the tracker on one (sequence, mode) pair and returns
// the aggregate result. limitFrames > 0 caps replay to the first N
// reference frames, which is needed for the with_embs mode because its
// embedding fixture only covers the first 10 frames.
func runParity(t *testing.T, seq, mode string, limitFrames int) parityResult {
	t.Helper()
	withReID := mode == "with_embs"

	ref := loadReference(t, filepath.Join(
		"testdata", "reference", fmt.Sprintf("%s_%s.json", seq, mode)))
	detsRows := loadRows(t, filepath.Join("testdata", "dets", seq+".json"))

	var embRows [][]float64
	if withReID {
		embRows = loadRows(t, filepath.Join("testdata", "embs", seq+".json"))
	}

	meta, ok := imgSize[seq]
	if !ok {
		t.Fatalf("unknown sequence %q", seq)
	}
	tr := newParityTracker(withReID)
	tr.SetImageSize(meta.W, meta.H)

	perFrame := make(map[int][]int, len(detsRows))
	for i, row := range detsRows {
		fid := int(row[0])
		perFrame[fid] = append(perFrame[fid], i)
	}

	// The embedding fixture only covers frames that appear in the reference,
	// so walk the detections in row order and pair each one up against the
	// next available embedding row. This gives a detRowIdx → embRowIdx
	// mapping that can be looked up per-detection below.
	embIdx := make(map[int]int)
	if withReID {
		next := 0
		for rowIdx, row := range detsRows {
			if !refHasFrame(&ref, int(row[0])) {
				continue
			}
			if next >= len(embRows) {
				break
			}
			embIdx[rowIdx] = next
			next++
		}
		if next != len(embRows) {
			t.Fatalf("emb/det row mismatch for %s: mapped %d, emb rows %d",
				seq, next, len(embRows))
		}
	}

	var res parityResult
	refToGoFirst := map[int]int{}

	for i, refFr := range ref.Frames {
		if limitFrames > 0 && i >= limitFrames {
			break
		}
		res.frames++
		fid := refFr.FrameID
		idxs := perFrame[fid]

		frameDets := make([]*parityDet, len(idxs))
		for j, rowIdx := range idxs {
			row := detsRows[rowIdx]
			d := &parityDet{
				xyxy:  [4]float64{row[1], row[2], row[3], row[4]},
				score: row[5],
				cls:   int(row[6]),
				idx:   j,
			}
			if withReID {
				ei, ok := embIdx[rowIdx]
				if !ok {
					t.Fatalf("frame %d rowIdx %d missing from emb map", fid, rowIdx)
				}
				d.emb = embRows[ei]
			}
			frameDets[j] = d
		}

		goOut, _ := tr.Update(frameDets)

		goByDet := make(map[int]Track[*parityDet], len(goOut))
		for _, g := range goOut {
			ext := g.(TrackExt[*parityDet])
			goByDet[ext.GetDetID()] = g
		}
		refByDet := make(map[int][]float64, len(refFr.Tracks))
		for _, row := range refFr.Tracks {
			if len(row) < 8 {
				continue
			}
			refByDet[int(row[7])] = row
		}

		res.refTotal += len(refByDet)
		res.goTotal += len(goByDet)
		if len(refByDet) != len(goByDet) {
			res.countMismatch++
		}

		for detID, refRow := range refByDet {
			g, ok := goByDet[detID]
			if !ok {
				continue
			}
			res.matchedDetIDs++
			xyxy := g.GetXYXY()
			for k := range 4 {
				a := math.Abs(xyxy[k] - refRow[k])
				if a > res.maxAbs {
					res.maxAbs = a
				}
			}
			goID := int(g.GetID())
			refID := int(refRow[4])
			if prev, seen := refToGoFirst[refID]; !seen {
				refToGoFirst[refID] = goID
			} else if prev != goID {
				res.idSwitches++
			}
		}
	}
	return res
}

func refHasFrame(r *refFile, fid int) bool {
	for _, fr := range r.Frames {
		if fr.FrameID == fid {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------

// maxBBoxAbs caps the per-frame ℓ∞ bbox error. The measured ceiling across
// all seven sequences in both modes is ~7e-5 — bounded by float32 storage
// upstream. Roughly 3× headroom leaves room for platform-dependent rounding.
const maxBBoxAbs = 2e-4

// ---------------------------------------------------------------------------
// Test entry points
// ---------------------------------------------------------------------------

func TestParityNoEmbs(t *testing.T) {
	for _, seq := range []string{
		"MOT17-02", "MOT17-04", "MOT17-05",
		"MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13",
	} {
		t.Run(seq, func(t *testing.T) {
			res := runParity(t, seq, "no_embs", 0)
			checkParity(t, res)
		})
	}
}

func TestParityWithEmbs(t *testing.T) {
	// Only MOT17-09 is shipped in-repo for the with_embs path — the full
	// per-sequence embedding file would exceed ~50 MB each. The fixture is
	// capped at the first 10 frames; the reference JSON is truncated to
	// match.
	if testing.Short() {
		t.Skip("skipping with_embs parity in -short mode (loads 9 MB embedding fixture)")
	}
	res := runParity(t, "MOT17-09", "with_embs", 10)
	checkParity(t, res)
}

func checkParity(t *testing.T, res parityResult) {
	t.Helper()
	t.Logf("frames=%d ref_total=%d go_total=%d matched=%d maxAbs=%.3g idSwitches=%d",
		res.frames, res.refTotal, res.goTotal, res.matchedDetIDs,
		res.maxAbs, res.idSwitches)
	if res.countMismatch != 0 {
		t.Errorf("per-frame track count diverged on %d frames", res.countMismatch)
	}
	if res.maxAbs > maxBBoxAbs {
		t.Errorf("bbox ℓ∞ = %.3g exceeds tolerance %.3g", res.maxAbs, maxBBoxAbs)
	}
	if res.idSwitches != 0 {
		t.Errorf("id stability broke: %d id switches (ref_id → different go_id later)", res.idSwitches)
	}
}
