// Package embedding implements the appearance-embedding update rules used
// to fuse a new observation's embedding into a track's stored embedding.
package embedding

import "math"

// Mode selects the update rule used by Handler.
type Mode uint8

const (
	// ModeEMA blends the stored embedding with the (normalized) new one via
	// a confidence-weighted exponential moving average. This is the
	// recommended setting: it smooths over single-frame appearance jitter
	// and down-weights low-confidence observations.
	ModeEMA Mode = iota
	// ModeLast overwrites the stored embedding with the normalized new one
	// every time. Useful when appearance changes faster than the EMA can
	// track, or as a debugging baseline.
	ModeLast
)

// Handler applies the configured update rule to a track's embedding given
// a matched detection's embedding. In ModeEMA, the fusion rule is
//
//	new_norm = new / ‖new‖₂
//	α        = (1 - EMAAlpha) · conf      (α = 1 - EMAAlpha when conf < 0)
//	out      = prev · (1 - α) + new_norm · α
//
// The result is deliberately not re-normalized: the distance code is
// robust to drift away from unit length, and re-normalizing each step
// would subtly amplify numerical noise.
type Handler struct {
	Mode     Mode
	EMAAlpha float64
}

// UpdateRow fuses (prev, next) into out in place. out may alias prev. The
// next slice is normalized before use, but is not mutated. A negative conf
// takes the "no-confidence" branch (α = 1 - EMAAlpha, applied uniformly).
//
// The arithmetic is carried out in float32 intentionally. The upstream
// embedding extractor stores its outputs in float32, and widening every
// intermediate to float64 diverges from that precision just enough that a
// track's pure embedding crosses the re-identification threshold after
// ~70 EMA steps. Matching the float32 step-by-step rounding keeps the
// long-term numerical behavior stable.
func (h Handler) UpdateRow(out, prev, next []float64, conf float64) {
	if len(out) != len(next) || len(prev) != len(next) {
		panic("embedding: UpdateRow length mismatch")
	}

	var norm32 float32
	for _, x := range next {
		x32 := float32(x)
		norm32 += x32 * x32
	}
	norm32 = float32(math.Sqrt(float64(norm32)))

	switch h.Mode {
	case ModeLast:
		if norm32 == 0 {
			copy(out, next)
			return
		}
		for i, v := range next {
			out[i] = float64(float32(v) / norm32)
		}
	default:
		var alpha32 float32
		if conf < 0 {
			alpha32 = float32(1 - h.EMAAlpha)
		} else {
			alpha32 = float32(1-h.EMAAlpha) * float32(conf)
		}
		oneMinusAlpha32 := 1 - alpha32
		if norm32 == 0 {
			// No new direction to blend in; the previous embedding is kept
			// but damped by (1 - α), matching the limit of the EMA formula
			// as ‖next‖ → 0.
			for i, p := range prev {
				out[i] = float64(float32(p) * oneMinusAlpha32)
			}
			return
		}
		for i := range next {
			nextN := float32(next[i]) / norm32
			a := nextN * alpha32
			b := float32(prev[i]) * oneMinusAlpha32
			out[i] = float64(a + b)
		}
	}
}

// UpdateBatch applies UpdateRow to every row. A nil confs slice selects
// the no-confidence branch for every row.
func (h Handler) UpdateBatch(out, prev, next [][]float64, confs []float64) {
	if len(out) != len(next) || len(prev) != len(next) {
		panic("embedding: UpdateBatch outer length mismatch")
	}
	for i := range next {
		conf := -1.0
		if confs != nil {
			conf = confs[i]
		}
		h.UpdateRow(out[i], prev[i], next[i], conf)
	}
}

// NormalizeRow writes the L2-normalized src into dst. dst may alias src.
// Zero-norm input is copied through unchanged, since there is no canonical
// direction to pick in that case.
func NormalizeRow(dst, src []float64) {
	if len(dst) != len(src) {
		panic("embedding: NormalizeRow length mismatch")
	}
	norm := 0.0
	for _, x := range src {
		norm += x * x
	}
	if norm == 0 {
		copy(dst, src)
		return
	}
	inv := 1.0 / math.Sqrt(norm)
	for i, x := range src {
		dst[i] = float64(float32(x * inv))
	}
}
