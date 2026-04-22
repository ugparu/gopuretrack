package kalman

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func puretrackFilter() *Filter {
	return New(Config{
		StdProcessPos:    0.05,
		StdProcessVel:    0.00625,
		StdMeasurement:   0.05,
		InitPosCovFactor: 2.0,
		InitVelCovFactor: 10.0,
	})
}

// TestInitiate pins the birth mean and covariance diagonal for two known
// measurements so any change to the initialization formula is caught.
func TestInitiate(t *testing.T) {
	f := puretrackFilter()
	measurements := [][4]float64{
		{10.0, 45.0, 7.8, 90.0},
		{0.7, 9.004, 78.98, -1.7},
	}
	means := make([][8]float64, 2)
	covs := make([][64]float64, 2)
	f.MultiInitiate(measurements, means, covs)

	require.InDeltaSlice(t,
		[]float64{10.0, 45.0, 7.8, 90.0, 0, 0, 0, 0}, means[0][:], 1e-12)
	require.InDeltaSlice(t,
		[]float64{0.7, 9.004, 78.98, -1.7, 0, 0, 0, 0}, means[1][:], 1e-12)

	diag := func(c [64]float64) []float64 {
		d := make([]float64, 8)
		for i := range 8 {
			d[i] = c[i*8+i]
		}
		return d
	}
	require.InDeltaSlice(t, []float64{
		0.6084, 81.0, 0.6084, 81.0,
		0.23765624999999999, 31.640625, 0.23765624999999999, 31.640625,
	}, diag(covs[0]), 1e-12)
	require.InDeltaSlice(t, []float64{
		62.37840400000001, 0.028900000000000006, 62.37840400000001, 0.028900000000000006,
		24.366564062500004, 0.011289062499999999, 24.366564062500004, 0.011289062499999999,
	}, diag(covs[1]), 1e-12)
}

// TestPredict pins the full 8×8 covariance after one predict step, which
// exercises both F · P · Fᵀ and the added process-noise diagonal.
func TestPredict(t *testing.T) {
	f := puretrackFilter()
	measurements := [][4]float64{
		{10.0, 45.0, 7.8, 90.0},
		{0.7, 9.004, 78.98, -1.7},
	}
	means := make([][8]float64, 2)
	covs := make([][64]float64, 2)
	f.MultiInitiate(measurements, means, covs)
	f.MultiPredict(means, covs)

	// Velocities are still zero at birth, so the mean should not move.
	require.InDeltaSlice(t,
		[]float64{10.0, 45.0, 7.8, 90.0, 0, 0, 0, 0}, means[0][:], 1e-12)

	expectedCov0 := []float64{
		0.9981562500000001, 0, 0, 0, 0.23765624999999999, 0, 0, 0,
		0, 132.890625, 0, 0, 0, 31.640625, 0, 0,
		0, 0, 0.9981562500000001, 0, 0, 0, 0.23765624999999999, 0,
		0, 0, 0, 132.890625, 0, 0, 0, 31.640625,
		0.23765624999999999, 0, 0, 0, 0.2400328125, 0, 0, 0,
		0, 31.640625, 0, 0, 0, 31.95703125, 0, 0,
		0, 0, 0.23765624999999999, 0, 0, 0, 0.2400328125, 0,
		0, 0, 0, 31.640625, 0, 0, 0, 31.95703125,
	}
	require.InDeltaSlice(t, expectedCov0, covs[0][:], 1e-10)
}

// TestProjectWithNSA checks that the measurement-noise diagonal scales by
// (1 - conf)² and falls back to plain R when conf is zero or confs is nil.
func TestProjectWithNSA(t *testing.T) {
	f := puretrackFilter()
	measurements := [][4]float64{
		{10.0, 45.0, 7.8, 90.0},
		{0.7, 9.004, 78.98, -1.7},
	}
	means := make([][8]float64, 2)
	covs := make([][64]float64, 2)
	f.MultiInitiate(measurements, means, covs)
	f.MultiPredict(means, covs)

	confs := []float64{0.8, 0.8}
	_, projCovs := f.MultiProject(means, covs, confs)

	require.InDeltaSlice(t, []float64{
		1.00424025, 0, 0, 0,
		0, 133.700625, 0, 0,
		0, 0, 1.00424025, 0,
		0, 0, 0, 133.700625,
	}, projCovs[0][:], 1e-10)
	require.InDeltaSlice(t, []float64{
		102.96335310250002, 0, 0, 0,
		0, 0.047703062500000004, 0, 0,
		0, 0, 102.96335310250002, 0,
		0, 0, 0, 0.047703062500000004,
	}, projCovs[1][:], 1e-10)

	_, projCovs0 := f.MultiProject(means, covs, []float64{0, 0})
	require.InDeltaSlice(t, []float64{
		1.15025625, 0, 0, 0,
		0, 153.140625, 0, 0,
		0, 0, 1.15025625, 0,
		0, 0, 0, 153.140625,
	}, projCovs0[0][:], 1e-10)

	// nil confs must behave identically to an all-zero conf slice.
	_, projCovsNil := f.MultiProject(means, covs, nil)
	require.InDeltaSlice(t, projCovs0[0][:], projCovsNil[0][:], 1e-12)
}

// TestUpdateWithNSA drives one full correction step with conf=0.8 and
// pins the posterior mean and covariance.
func TestUpdateWithNSA(t *testing.T) {
	f := puretrackFilter()
	measurements := [][4]float64{
		{10.0, 45.0, 7.8, 90.0},
		{0.7, 9.004, 78.98, -1.7},
	}
	means := make([][8]float64, 2)
	covs := make([][64]float64, 2)
	f.MultiInitiate(measurements, means, covs)
	f.MultiPredict(means, covs)

	meas2 := [][4]float64{
		{70.0, 43.0, 0.8, 67.0},
		{1.7, 5.004, 78.0, -1.5},
	}
	confs := []float64{0.8, 0.8}
	f.MultiUpdate(means, covs, meas2, confs)

	require.InDeltaSlice(t, []float64{
		69.63650132525558, 43.01211662249148, 0.8424081787201816, 67.13934115865203,
		14.199166982203709, -0.47330556607345703, -1.6565694812570995, -5.443014009844756,
	}, means[0][:], 1e-8)
	require.InDeltaSlice(t, []float64{
		1.6939416887542598, 5.02823324498296, 78.00593714502082, -1.501211662249148,
		0.23665278303672851, -0.9466111321469138, -0.2319197273759949, 0.047330556607345683,
	}, means[1][:], 1e-8)

	expectedCov0 := []float64{
		0.006047141234380882, 0, 0, 0, 0.0014397955319954414, 0, 0, 0,
		0, 0.8050927678909545, 0, 0, 0, 0.19168875425975074, 0, 0,
		0, 0, 0.006047141234380882, 0, 0, 0, 0.0014397955319954414, 0,
		0, 0, 0, 0.8050927678909545, 0, 0, 0, 0.19168875425975074,
		0.0014397955319954414, 0, 0, 0, 0.1837907995314275, 0, 0, 0,
		0, 0.19168875425974718, 0, 0, 0, 24.46918928672851, 0, 0,
		0, 0, 0.0014397955319954414, 0, 0, 0, 0.1837907995314275, 0,
		0, 0, 0, 0.19168875425974718, 0, 0, 0, 24.46918928672851,
	}
	require.InDeltaSlice(t, expectedCov0, covs[0][:], 1e-9)
}

// TestEmptyBatch verifies that every entry point is a no-op on empty
// input — a property the pipeline relies on to avoid special-casing frames
// with no tracks or no detections.
func TestEmptyBatch(t *testing.T) {
	f := puretrackFilter()
	var empty [][4]float64
	means := [][8]float64{}
	covs := [][64]float64{}
	f.MultiInitiate(empty, means, covs)
	f.MultiPredict(means, covs)
	f.MultiUpdate(means, covs, empty, nil)
	pm, pc := f.MultiProject(means, covs, nil)
	require.Len(t, pm, 0)
	require.Len(t, pc, 0)
}
