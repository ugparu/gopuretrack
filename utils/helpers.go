// Package utils is a collection of math and geometry helpers shared by
// the tracker pipeline: bounding-box format conversions, batched IoU,
// cosine-distance matrices, visibility ratios, and L2 normalization.
package utils

import (
	"math"
	"slices"
)

// XYXY2XYWH converts a corner box (x1, y1, x2, y2) to a center-plus-size
// box (xc, yc, w, h).
func XYXY2XYWH(b [4]float64) [4]float64 {
	return [4]float64{
		(b[0] + b[2]) / 2,
		(b[1] + b[3]) / 2,
		b[2] - b[0],
		b[3] - b[1],
	}
}

// XYWH2XYXY is the inverse of XYXY2XYWH.
func XYWH2XYXY(b [4]float64) [4]float64 {
	hw := b[2] / 2
	hh := b[3] / 2
	return [4]float64{b[0] - hw, b[1] - hh, b[0] + hw, b[1] + hh}
}

// resize2D grows or truncates m to exactly rows×cols, reusing both the
// outer backing array and each row's backing array when capacity permits.
// Pass a nil m to allocate fresh.
func resize2D(m [][]float64, rows, cols int) [][]float64 {
	if cap(m) < rows {
		m = make([][]float64, rows)
	} else {
		m = m[:rows]
	}
	for i := range rows {
		if cap(m[i]) < cols {
			m[i] = make([]float64, cols)
		} else {
			m[i] = m[i][:cols]
		}
	}
	return m
}

// IoUBatchXYWH writes the intersection-over-union matrix between two sets
// of (xc, yc, w, h) boxes into out, resizing out to len(a) × len(b). Pass
// nil for out to get a fresh allocation; hot-path callers should hold a
// buffer and pass it in. Returns the (possibly reallocated) out slice.
func IoUBatchXYWH(out [][]float64, a, b [][4]float64) [][]float64 {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}
	out = resize2D(out, len(a), len(b))
	for i := range a {
		ax, ay, aw, ah := a[i][0], a[i][1], a[i][2], a[i][3]
		aHalfW, aHalfH := aw/2, ah/2
		aArea := aw * ah
		row := out[i]
		for j := range b {
			bx, by, bw, bh := b[j][0], b[j][1], b[j][2], b[j][3]
			bHalfW, bHalfH := bw/2, bh/2
			left := max(ax-aHalfW, bx-bHalfW)
			top := max(ay-aHalfH, by-bHalfH)
			right := min(ax+aHalfW, bx+bHalfW)
			bottom := min(ay+aHalfH, by+bHalfH)
			iw := max(0.0, right-left)
			ih := max(0.0, bottom-top)
			inter := iw * ih
			row[j] = inter / (aArea + bw*bh - inter)
		}
	}
	return out
}

// EmbeddingDistance returns a cosine-distance matrix scaled into [0, 1].
// The inputs do not have to be unit-length: each row is normalized
// internally, so a track whose EMA-updated embedding has drifted away from
// unit length still produces a well-behaved distance.
//
// The result has shape len(tracksEmbs) × len(detsEmbs).
func EmbeddingDistance(tracksEmbs, detsEmbs [][]float64) [][]float64 {
	rows := len(tracksEmbs)
	cols := len(detsEmbs)
	if rows == 0 || cols == 0 {
		return nil
	}
	dim := len(tracksEmbs[0])

	trackNorms := make([]float64, rows)
	for i, t := range tracksEmbs {
		var n float64
		for _, x := range t {
			n += x * x
		}
		trackNorms[i] = math.Sqrt(n)
	}
	detNorms := make([]float64, cols)
	for j, d := range detsEmbs {
		var n float64
		for _, x := range d {
			n += x * x
		}
		detNorms[j] = math.Sqrt(n)
	}

	out := make([][]float64, rows)
	for i := range rows {
		out[i] = make([]float64, cols)
		ti := tracksEmbs[i]
		tn := trackNorms[i]
		for j := range cols {
			dj := detsEmbs[j]
			dot := 0.0
			for k := range dim {
				dot += ti[k] * dj[k]
			}
			denom := tn * detNorms[j]
			var d float64
			if denom == 0 {
				d = 1
			} else {
				d = 1 - dot/denom
			}
			if d < 0 {
				d = 0
			}
			out[i][j] = d * 0.5
		}
	}
	return out
}

// FuseScore folds per-column detection confidences into an association
// cost matrix: cost'[i, j] = 1 - (1 - cost[i, j]) · confs[j]. A confident
// detection (confs[j] near 1) preserves its cost row; a low-confidence
// detection is pushed toward 1, making matches against it less competitive.
//
// Writes into out, resizing to match costs. Safe to pass `out == costs`
// for in-place fusion: each cell is read before being written. Pass nil
// for out to get a fresh allocation.
func FuseScore(out, costs [][]float64, confs []float64) [][]float64 {
	if len(costs) == 0 || len(confs) == 0 {
		return nil
	}
	rows := len(costs)
	cols := len(costs[0])
	out = resize2D(out, rows, cols)
	for i := range rows {
		src := costs[i]
		dst := out[i]
		for j := range cols {
			dst[j] = 1 - (1-src[j])*confs[j]
		}
	}
	return out
}

// BoxVisibilityRatiosBatch returns, for each input box j, the largest
// fraction of j's area covered by any other box in the input. A value near
// zero means j is effectively unoccluded; a value near one means another
// box covers it almost entirely.
//
// Boxes are in (xc, yc, w, h). The result has length len(boxes).
func BoxVisibilityRatiosBatch(boxes [][4]float64) []float64 {
	n := len(boxes)
	out := make([]float64, n)
	if n <= 1 {
		return out
	}
	for j := range n {
		bj := boxes[j]
		areaJ := bj[2] * bj[3]
		hwJ, hhJ := bj[2]/2, bj[3]/2
		maxOverlap := 0.0
		for i := range n {
			if i == j {
				continue
			}
			bi := boxes[i]
			hwI, hhI := bi[2]/2, bi[3]/2
			left := max(bi[0]-hwI, bj[0]-hwJ)
			top := max(bi[1]-hhI, bj[1]-hhJ)
			right := min(bi[0]+hwI, bj[0]+hwJ)
			bottom := min(bi[1]+hhI, bj[1]+hhJ)
			iw := max(0.0, right-left)
			ih := max(0.0, bottom-top)
			o := (iw * ih) / areaJ
			if o > maxOverlap {
				maxOverlap = o
			}
		}
		out[j] = maxOverlap
	}
	return out
}

// L2Normalize scales v to unit length in place. Degenerate input
// (zero-norm or producing a non-finite reciprocal norm) is left unchanged,
// which is the only safe choice when there is no canonical direction to
// pick.
func L2Normalize(v []float64) {
	s := 0.0
	for _, x := range v {
		s += x * x
	}
	if s == 0 {
		return
	}
	inv := 1.0 / math.Sqrt(s)
	if math.IsInf(inv, 0) || math.IsNaN(inv) {
		return
	}
	for i := range v {
		v[i] *= inv
	}
}

// SearchInMap returns the keys whose values satisfy the predicate f.
// Iteration order follows the map's own, so callers that need a stable
// order must sort the result themselves.
func SearchInMap[K comparable, V any](pool map[K]V, f func(V) bool) []K {
	var keys []K
	for k, v := range pool {
		if f(v) {
			keys = append(keys, k)
		}
	}
	return keys
}

// AdjustSliceSize returns a slice of the requested length whose backing
// array has capacity for at least size elements, reusing the input's
// allocation when possible.
func AdjustSliceSize[T any](slice []T, size int) []T {
	return slices.Grow(slice, size)[:size]
}
