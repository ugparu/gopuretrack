package embedding

import (
	"math"
	"testing"

	"github.com/stretchr/testify/require"
)

// The expected values below are taken from the EMA formula:
//
//	new_norm = new / ‖new‖₂
//	α        = (1 - EMAAlpha) · conf      (α = 1 - EMAAlpha when conf < 0)
//	result   = prev · (1 - α) + new_norm · α
//
// The result is deliberately not re-normalized.
func TestHandlerUpdateBatch_EMA_WithConf(t *testing.T) {
	prev := [][]float64{
		{0.6, 0.8, 0.0},
		{0.26726124, 0.53452248, 0.80178373},
	}
	newE := [][]float64{
		{3.0, 4.0, 0.0},
		{1.0, 0.0, 1.0},
	}
	confs := []float64{0.9, 0.7}

	out := [][]float64{make([]float64, 3), make([]float64, 3)}
	h := Handler{Mode: ModeEMA, EMAAlpha: 0.9}
	h.UpdateBatch(out, prev, newE, confs)

	expected := [][]float64{
		{0.6, 0.8, 0.0},
		{0.2980504278830583, 0.4971059064, 0.7951563435830584},
	}
	require.InDeltaSlice(t, expected[0], out[0], 1e-6)
	require.InDeltaSlice(t, expected[1], out[1], 1e-6)
}

func TestHandlerUpdateBatch_EMA_NoConf(t *testing.T) {
	prev := [][]float64{
		{0.6, 0.8, 0.0},
		{0.26726124, 0.53452248, 0.80178373},
	}
	newE := [][]float64{
		{3.0, 4.0, 0.0},
		{1.0, 0.0, 1.0},
	}

	out := [][]float64{make([]float64, 3), make([]float64, 3)}
	h := Handler{Mode: ModeEMA, EMAAlpha: 0.9}
	h.UpdateBatch(out, prev, newE, nil)

	expected := [][]float64{
		{0.6, 0.8, 0.0},
		{0.3112457941186547, 0.48107023200000004, 0.7923160351186548},
	}
	require.InDeltaSlice(t, expected[0], out[0], 1e-6)
	require.InDeltaSlice(t, expected[1], out[1], 1e-6)
}

func TestHandlerUpdateBatch_Last(t *testing.T) {
	prev := [][]float64{
		{9.9, 9.9, 9.9}, // should be ignored
		{1.0, 1.0, 1.0},
	}
	newE := [][]float64{
		{3.0, 4.0, 0.0},
		{1.0, 0.0, 1.0},
	}

	out := [][]float64{make([]float64, 3), make([]float64, 3)}
	h := Handler{Mode: ModeLast}
	h.UpdateBatch(out, prev, newE, []float64{0.5, 0.5})

	expected := [][]float64{
		{0.6, 0.8, 0.0},
		{0.7071067811865475, 0.0, 0.7071067811865475},
	}
	require.InDeltaSlice(t, expected[0], out[0], 1e-6)
	require.InDeltaSlice(t, expected[1], out[1], 1e-6)
}

// When the new embedding has zero norm there is no direction to blend in,
// so the formula reduces to prev · (1 - α).
func TestHandlerUpdateRow_EMA_ZeroNew(t *testing.T) {
	prev := []float64{0.6, 0.8, 0.0}
	newE := []float64{0.0, 0.0, 0.0}
	out := make([]float64, 3)

	h := Handler{Mode: ModeEMA, EMAAlpha: 0.9}
	h.UpdateRow(out, prev, newE, 0.8)

	expected := []float64{0.552, 0.7360000000000001, 0.0}
	require.InDeltaSlice(t, expected, out, 1e-6)
}

// UpdateRow must allow out and prev to be the same slice so that callers
// can EMA a track's embedding in place without allocating a scratch buffer.
func TestHandlerUpdateRow_Alias(t *testing.T) {
	buf := []float64{0.6, 0.8, 0.0}
	newE := []float64{3.0, 4.0, 0.0}

	h := Handler{Mode: ModeEMA, EMAAlpha: 0.9}
	h.UpdateRow(buf, buf, newE, 0.9)

	require.InDeltaSlice(t, []float64{0.6, 0.8, 0.0}, buf, 1e-6)
}

func TestHandlerUpdateRow_LengthMismatch(t *testing.T) {
	h := Handler{Mode: ModeEMA, EMAAlpha: 0.9}
	require.Panics(t, func() {
		h.UpdateRow(make([]float64, 2), []float64{1, 2, 3}, []float64{1, 2, 3}, 1.0)
	})
}

func TestNormalizeRow(t *testing.T) {
	dst := make([]float64, 2)
	NormalizeRow(dst, []float64{3, 4})
	require.InDeltaSlice(t, []float64{0.6, 0.8}, dst, 1e-6)

	zero := []float64{0, 0, 0}
	NormalizeRow(zero, zero)
	require.Equal(t, []float64{0, 0, 0}, zero)

	dst3 := make([]float64, 3)
	NormalizeRow(dst3, []float64{1.0, 2.0, 2.0})
	n := 0.0
	for _, x := range dst3 {
		n += x * x
	}
	require.InDelta(t, 1.0, math.Sqrt(n), 1e-6)
}
