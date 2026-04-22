package utils

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestXYXYXYWHRoundtrip(t *testing.T) {
	xyxy := [4]float64{10, 20, 110, 220}
	xywh := XYXY2XYWH(xyxy)
	require.InDelta(t, 60.0, xywh[0], 1e-12)
	require.InDelta(t, 120.0, xywh[1], 1e-12)
	require.InDelta(t, 100.0, xywh[2], 1e-12)
	require.InDelta(t, 200.0, xywh[3], 1e-12)

	back := XYWH2XYXY(xywh)
	require.InDeltaSlice(t, xyxy[:], back[:], 1e-12)
}

func TestEmbeddingDistance(t *testing.T) {
	tracksEmbs := [][]float64{
		{0.26726124, 0.53452248, 0.80178373},
		{0.45584231, 0.56980288, 0.68376346},
	}
	detsEmbs := [][]float64{
		{0.11624764, 0.34874292, 0.92998111},
		{0.42426407, 0.56568542, 0.707106786},
	}
	expected := [][]float64{
		{0.018438430090733593, 0.008646185278256502},
		{0.056203794079371705, 0.00038976239576438143},
	}

	out := EmbeddingDistance(tracksEmbs, detsEmbs)
	require.Len(t, out, 2)
	require.Len(t, out[0], 2)
	require.InDeltaSlice(t, expected[0], out[0], 1e-8)
	require.InDeltaSlice(t, expected[1], out[1], 1e-8)

	// Rectangular shape (2 × 3) exercises the row-vs-column indexing and
	// catches the easy-to-make mistake of sizing the output off the track
	// count alone.
	detsEmbs3 := append([][]float64(nil), detsEmbs...)
	detsEmbs3 = append(detsEmbs3, []float64{1, 0, 0})
	out = EmbeddingDistance(tracksEmbs, detsEmbs3)
	require.Len(t, out, 2)
	require.Len(t, out[0], 3)

	require.Nil(t, EmbeddingDistance(nil, detsEmbs))
	require.Nil(t, EmbeddingDistance(tracksEmbs, nil))
}

func TestFuseScore(t *testing.T) {
	costMatrix := [][]float64{
		{0.78, 0.2},
		{0.9, 1.0},
	}
	confs := []float64{0.8, 0.9}
	expected := [][]float64{
		{0.824, 0.28},
		{0.92, 1.0},
	}

	out := FuseScore(nil, costMatrix, confs)
	require.InDeltaSlice(t, expected[0], out[0], 1e-10)
	require.InDeltaSlice(t, expected[1], out[1], 1e-10)

	require.Nil(t, FuseScore(nil, nil, confs))
	require.Nil(t, FuseScore(nil, costMatrix, nil))

	// In-place form: passing costMatrix as both out and costs writes the
	// fused values over the input.
	costCopy := [][]float64{
		{0.78, 0.2},
		{0.9, 1.0},
	}
	FuseScore(costCopy, costCopy, confs)
	require.InDeltaSlice(t, expected[0], costCopy[0], 1e-10)
	require.InDeltaSlice(t, expected[1], costCopy[1], 1e-10)
}

func TestIoUBatchXYWH(t *testing.T) {
	b1 := [][4]float64{
		{50, 50, 100, 100},
		{30, 30, 50, 50},
		{200, 200, 150, 150},
		{400, 400, 100, 100},
	}
	b2 := [][4]float64{
		{60, 60, 80, 80},
		{0, 0, 100, 100},
		{450, 450, 50, 50},
	}
	expected := [][]float64{
		{0.64, 0.14285714285714285, 0.0},
		{0.15960912052117263, 0.19331742243436753, 0.0},
		{0.0, 0.0, 0.0},
		{0.0, 0.0, 0.05263157894736842},
	}

	out := IoUBatchXYWH(nil, b1, b2)
	require.Len(t, out, 4)
	for i := range expected {
		require.InDeltaSlice(t, expected[i], out[i], 1e-12)
	}

	require.Nil(t, IoUBatchXYWH(nil, nil, b2))
	require.Nil(t, IoUBatchXYWH(nil, b1, nil))

	// Passing an existing buffer should reuse it and produce identical output.
	buf := IoUBatchXYWH(out, b1, b2)
	require.Len(t, buf, 4)
	for i := range expected {
		require.InDeltaSlice(t, expected[i], buf[i], 1e-12)
	}
}

func TestBoxVisibilityRatiosBatch(t *testing.T) {
	// Boxes 0, 1 and 3 mutually overlap; box 2 is isolated. The highest
	// overlap for box 0 comes from box 3: intersection 30×30 over area
	// 40×40 = 0.5625.
	boxes := [][4]float64{
		{100, 100, 40, 40},
		{120, 100, 40, 40},
		{500, 500, 20, 20},
		{110, 110, 40, 40},
	}
	expected := []float64{0.5625, 0.5625, 0.0, 0.5625}

	out := BoxVisibilityRatiosBatch(boxes)
	require.InDeltaSlice(t, expected, out, 1e-12)

	// With a single input there is nothing to overlap with, so the result
	// must be zero rather than undefined.
	require.Equal(t, []float64{0}, BoxVisibilityRatiosBatch([][4]float64{{10, 10, 5, 5}}))
	require.Equal(t, []float64{}, BoxVisibilityRatiosBatch(nil))
}

func TestL2Normalize(t *testing.T) {
	v := []float64{3, 4}
	L2Normalize(v)
	require.InDeltaSlice(t, []float64{0.6, 0.8}, v, 1e-12)

	zero := []float64{0, 0, 0}
	L2Normalize(zero)
	require.Equal(t, []float64{0, 0, 0}, zero)
}
