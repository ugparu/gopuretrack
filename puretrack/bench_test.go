package puretrack

import (
	"path/filepath"
	"testing"
)

// Each benchmark iteration replays the whole sequence from a fresh tracker.
// ns/op reports full-replay time; the "frames/s" custom metric is the more
// interpretable throughput figure.

var benchSequences = []string{
	"MOT17-02", "MOT17-04", "MOT17-05",
	"MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13",
}

func BenchmarkTracker(b *testing.B) {
	for _, seq := range benchSequences {
		b.Run(seq, func(b *testing.B) {
			frames := loadSequenceFrames(b, seq, false, 0)
			meta := imgSize[seq]

			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				tr := newParityTracker(false)
				tr.SetImageSize(meta.W, meta.H)
				for _, fd := range frames {
					tr.Update(fd)
				}
			}
			b.StopTimer()
			b.ReportMetric(
				float64(b.N*len(frames))/b.Elapsed().Seconds(),
				"frames/s")
		})
	}
}

func BenchmarkTrackerWithEmbs(b *testing.B) {
	// Only MOT17-09's emb fixture ships in-repo (other sequences' embeddings
	// would exceed ~200 MB each). The full 262-frame sequence is replayed.
	const seq = "MOT17-09"
	frames := loadSequenceFrames(b, seq, true, 0)
	meta := imgSize[seq]

	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		tr := newParityTracker(true)
		tr.SetImageSize(meta.W, meta.H)
		for _, fd := range frames {
			tr.Update(fd)
		}
	}
	b.StopTimer()
	b.ReportMetric(
		float64(b.N*len(frames))/b.Elapsed().Seconds(),
		"frames/s")
}

// loadSequenceFrames builds, in reference-frame order, the per-frame slices
// of detections that the benchmark will stream through the tracker. withReID
// additionally attaches embeddings (only MOT17-09 ships one). limitFrames
// caps the replay length; 0 means the full sequence.
func loadSequenceFrames(tb testing.TB, seq string, withReID bool, limitFrames int) [][]*parityDet {
	tb.Helper()
	mode := "no_embs"
	if withReID {
		mode = "with_embs"
	}
	ref := loadReference(tb, filepath.Join(
		"testdata", "reference", seq+"_"+mode+".json"))
	detsRows := loadRows(tb, filepath.Join("testdata", "dets", seq+".json"))

	var embRows [][]float64
	if withReID {
		embRows = loadEmbRows(tb, filepath.Join("testdata", "embs", seq+".json"))
	}

	perFrame := make(map[int][]int, len(detsRows))
	for i, row := range detsRows {
		fid := int(row[0])
		perFrame[fid] = append(perFrame[fid], i)
	}

	// Same det→emb pairing logic as runParity: walk detections in row order
	// and consume emb rows one-by-one for each detection whose frame appears
	// in the reference.
	embIdx := make(map[int]int)
	if withReID {
		refFrames := make(map[int]struct{}, len(ref.Frames))
		for _, fr := range ref.Frames {
			refFrames[fr.FrameID] = struct{}{}
		}
		next := 0
		for rowIdx, row := range detsRows {
			if _, ok := refFrames[int(row[0])]; !ok {
				continue
			}
			if next >= len(embRows) {
				break
			}
			embIdx[rowIdx] = next
			next++
		}
		if next != len(embRows) {
			tb.Fatalf("emb/det row mismatch for %s: mapped %d, emb rows %d",
				seq, next, len(embRows))
		}
	}

	framesOut := make([][]*parityDet, 0, len(ref.Frames))
	for i, fr := range ref.Frames {
		if limitFrames > 0 && i >= limitFrames {
			break
		}
		idxs := perFrame[fr.FrameID]
		dets := make([]*parityDet, len(idxs))
		for j, rowIdx := range idxs {
			row := detsRows[rowIdx]
			d := &parityDet{
				xyxy:  [4]float64{row[1], row[2], row[3], row[4]},
				score: row[5],
				cls:   int(row[6]),
				idx:   j,
			}
			if withReID {
				if ei, ok := embIdx[rowIdx]; ok {
					d.emb = embRows[ei]
				}
			}
			dets[j] = d
		}
		framesOut = append(framesOut, dets)
	}
	return framesOut
}
